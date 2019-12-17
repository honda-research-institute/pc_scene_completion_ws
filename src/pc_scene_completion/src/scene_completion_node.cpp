#include <ros/ros.h>
#include <scene_completion_node.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_rbf.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/pcl_config.h>
#include <time.h>

ros::Publisher cloud_pub;
typedef pcl::PointXYZ PointT;
typedef PointCloudT PointCloudT;
std::string cloud_target_topic;
bool detailed_mesh, extract_clusters;

SceneCompletionNode::SceneCompletionNode(ros::NodeHandle nh) :
    nh_(nh),
    reconfigure_server_(nh),
    as_(nh_, "SceneCompletion", boost::bind(&SceneCompletionNode::executeCB, this, _1), false),
    depth_cnn_client("/depth/object_completion", true),
    depth_tactile_cnn_client("/depth_tactile/object_completion", true),
    partial_client("/partial_object_completion", true),
    partial_mesh_count(0)

{
    nh.getParam("filtered_cloud_topic", filtered_cloud_topic);// /filter_pc
    nh.getParam("camera_frame", camera_frame);
    nh.getParam("world_frame", world_frame);
    nh.getParam("/topics/cloud_target_topic", cloud_target_topic);
    nh.getParam("/icp/voxel_leaf_size", voxel_leaf_size);
    nh.getParam("/flags/detailed_mesh", detailed_mesh);
    nh.getParam("/flags/extract_clusters", extract_clusters);

    // Set up dynamic reconfigure
    reconfigure_server_.setCallback(boost::bind(&SceneCompletionNode::reconfigure_cb, this, _1, _2));

    // Construct subscribers and publishers
    cloud_sub_ = nh.subscribe(filtered_cloud_topic, 1, &SceneCompletionNode::pcl_cloud_cb, this);
    if(detailed_mesh){
      cloud_pub = nh.advertise<sensor_msgs::PointCloud2>(cloud_target_topic, 1);
    }

    as_.start();

    ROS_INFO("SceneCompletionNode Initialized: ");
    std::cout << PCL_VERSION << std::endl;
    ROS_INFO_STREAM("filtered_cloud_topic: " << filtered_cloud_topic);
    ROS_INFO_STREAM("camera_frame: " << camera_frame);
    ROS_INFO_STREAM("world_frame: " << world_frame);
}


//!
//! \brief SceneCompletionNode::pcl_cloud_cb
//! \param points_msg
//!
//! This function takes a new point cloud message and addes it to the clouds_queue.
//! If the queue is already at its max size, then the oldest cloud is removed.
void SceneCompletionNode::pcl_cloud_cb(const sensor_msgs::PointCloud2ConstPtr &points_msg){
    // Lock the buffer mutex while we're capturing a new point cloud
    boost::mutex::scoped_lock buffer_lock(buffer_mutex_);


    ROS_INFO_STREAM("points_msg size = " <<  points_msg->width);
    // Convert to PCL cloud
    boost::shared_ptr<PointCloudT > cloud(new PointCloudT);
    pcl::fromROSMsg(*points_msg, *cloud);

    // Store the cloud
    clouds_queue_.push_back(cloud);

    // Increment the cloud index
    while(clouds_queue_.size() > (unsigned)n_clouds_per_recognition){
        clouds_queue_.pop_front();
    }
}

//update variables from config server
void SceneCompletionNode::reconfigure_cb(pc_scene_completion::SceneCompletionConfig &config, uint32_t level){
    //change any member vars here
    n_clouds_per_recognition = config.n_clouds_per_recognition;
    cluster_tolerance = config.cluster_tolerance;
    min_cluster_size = config.min_cluster_size;
    max_cluster_size = config.max_cluster_size;
}


void SceneCompletionNode::executeCB(const pc_pipeline_msgs::CompleteSceneGoalConstPtr & goal){

    ROS_INFO("received new CompleteSceneGoal");
    pc_pipeline_msgs::CompleteSceneResult result;

    ROS_INFO("Merging PointClouds");
    // merged set of captured pointclouds
    PointCloudT::Ptr cloud_full(new PointCloudT());

    // wait until n_clouds_per_recognition clouds have been received
    ros::Rate warn_rate(1.0);
    if(clouds_queue_.empty()){
       ROS_WARN("Pointcloud buffer is empty!");
       warn_rate.sleep();
    }

     // grab mutex so no one else touches cloud list
    boost::mutex::scoped_lock buffer_lock(buffer_mutex_);

    // merge all clouds together
    for(std::list<boost::shared_ptr<PointCloudT > >::const_iterator it = clouds_queue_.begin(); it != clouds_queue_.end(); ++it){
	     *cloud_full += *(*it);
    }

    if (extract_clusters){
      // extract clusters from cloud
      ROS_INFO("Extracting Clusters");
      pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
      tree->setInputCloud (cloud_full);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<PointT> ec;
      ec.setClusterTolerance (cluster_tolerance); // 2cm
      ec.setMinClusterSize (min_cluster_size);
      ec.setMaxClusterSize (max_cluster_size);
      ec.setSearchMethod (tree);
      ec.setInputCloud (cloud_full);
      ec.extract (cluster_indices);
      ROS_INFO_STREAM("no. of clusters found are: " << cluster_indices.size());

      ROS_INFO("Looking up %s to %s transform", world_frame.c_str(), camera_frame.c_str());
      tf::TransformListener listener;
      tf::StampedTransform transformMsg;
      Eigen::Matrix4f transformEigen = Eigen::Matrix4f::Identity ();

      listener.waitForTransform(world_frame, camera_frame, ros::Time(0), ros::Duration(3.0));
      listener.lookupTransform(world_frame, camera_frame, ros::Time(0), transformMsg);

      // transform from camera to world
      pcl_ros::transformAsMatrix(transformMsg, transformEigen);

      for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){

        // set of points corresponding to the visible region of a single object
        PointCloudT::Ptr cloud_cluster (new PointCloudT);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++){
            cloud_cluster->points.push_back (cloud_full->points[*pit]); //*
        }

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        geometry_msgs::PoseStamped poseStampedMsg;
        shape_msgs::Mesh meshMsg;
        sensor_msgs::PointCloud2 partialCloudMsg;

      	ROS_INFO("Calling point_cloud_to_mesh");
      	point_cloud_to_mesh(cloud_cluster, transformEigen, goal->object_completion_topic, meshMsg, poseStampedMsg, partialCloudMsg);

      	result.partial_views.push_back(partialCloudMsg);
      	result.meshes.push_back(meshMsg);
      	result.poses.push_back(poseStampedMsg);
      }
  } else{
      ROS_INFO("Not extracting clusters");
      ROS_INFO("Looking up %s to %s transform", world_frame.c_str(), camera_frame.c_str());
      tf::TransformListener listener;
      tf::StampedTransform transformMsg;
      Eigen::Matrix4f transformEigen = Eigen::Matrix4f::Identity ();

      listener.waitForTransform(world_frame, camera_frame, ros::Time(0), ros::Duration(3.0));
      listener.lookupTransform(world_frame, camera_frame, ros::Time(0), transformMsg);

      // transform from camera to world
      pcl_ros::transformAsMatrix(transformMsg, transformEigen);

      geometry_msgs::PoseStamped poseStampedMsg;
      shape_msgs::Mesh meshMsg;
      sensor_msgs::PointCloud2 partialCloudMsg;

      ROS_INFO("Calling point_cloud_to_mesh");
      point_cloud_to_mesh(cloud_full, transformEigen, goal->object_completion_topic, meshMsg, poseStampedMsg, partialCloudMsg);

      result.partial_views.push_back(partialCloudMsg);
      result.meshes.push_back(meshMsg);
      result.poses.push_back(poseStampedMsg);
  }
    as_.setSucceeded(result);
}


void SceneCompletionNode::point_cloud_to_mesh(PointCloudT::Ptr cloud,
                                              Eigen::Matrix4f transformEigen,
					                                    std::string object_completion_topic,
                                              shape_msgs::Mesh &meshMsg,
                                              geometry_msgs::PoseStamped  &poseStampedMsg,
                                              sensor_msgs::PointCloud2 &partialCloudMsg ){

    // convert pointcloud to ROS message
    pcl::PCLPointCloud2 partialCloudpc2CameraFrame;
    pcl::toPCLPointCloud2(*cloud, partialCloudpc2CameraFrame);
    sensor_msgs::PointCloud2 partialCloudMsgCameraFrame;
    pcl_conversions::fromPCL(partialCloudpc2CameraFrame, partialCloudMsgCameraFrame);

    // build up complete object goal
    pc_pipeline_msgs::CompletePartialCloudGoal goal;
    goal.partial_cloud = partialCloudMsgCameraFrame;

    actionlib::SimpleActionClient<pc_pipeline_msgs::CompletePartialCloudAction>* client;
    if (object_completion_topic == "depth"){
    	ROS_INFO_STREAM("object_completion_topic: " << object_completion_topic << " 0 " << std::endl);
    	client = &depth_cnn_client;
    }
    else if (object_completion_topic=="depth_tactile"){
    	ROS_INFO_STREAM("object_completion_topic: " << object_completion_topic << " 1 " << std::endl);
    	client = &depth_tactile_cnn_client;
    }
    else{
      ROS_INFO_STREAM("object_completion_topic: " << object_completion_topic << " 2 " << std::endl);
      client = &partial_client;
    }

    // send goal to complete object client and get result back
    ROS_INFO_STREAM("waitForServer()" << std::endl);
    client->waitForServer();
    ROS_INFO_STREAM("sendGoalAndWait()" << std::endl);
    client->sendGoalAndWait(goal);

    if(detailed_mesh){
      pc_pipeline_msgs::CompletePartialCloudResultConstPtr result = client->getResult();

      // mesh with points in the camera frame
      PointCloudT::Ptr meshVerticesCameraFrame(new PointCloudT());
      for(int i = 0; i < result->mesh.vertices.size(); i++){
          geometry_msgs::Point p = result->mesh.vertices.at(i);
          PointT pcl_point; // DO I need to do new for every point, or is += on a pointcloud a copy constructor
          pcl_point.x = p.x;
          pcl_point.y = p.y;
          pcl_point.z = p.z;
          meshVerticesCameraFrame->points.push_back(pcl_point);
      }

      // publish without RGB data
      PointCloudT::Ptr my_cloud_camera_frame(new PointCloudT);
      for(int i = 0; i < result->mesh.vertices.size(); i++){
          geometry_msgs::Point p = result->mesh.vertices.at(i);
          pcl::PointXYZ my_cloud; // DO I need to do new for every point, or is += on a pointcloud a copy constructor
          my_cloud.x = p.x;
          my_cloud.y = p.y;
          my_cloud.z = p.z;
          my_cloud_camera_frame->points.push_back(my_cloud);
      }

      // voxel grid downsampling filtering
      pcl::VoxelGrid<pcl::PointXYZ> sor;
      sor.setInputCloud(my_cloud_camera_frame);
      sor.setLeafSize(voxel_leaf_size, voxel_leaf_size, voxel_leaf_size);
      PointCloudT::Ptr my_cloud_filtered (new PointCloudT);
      sor.filter(*my_cloud_filtered);

      sensor_msgs::PointCloud2 cloud_cf;
      pcl::toROSMsg(*my_cloud_filtered, cloud_cf);
      cloud_cf.header.frame_id = camera_frame;
      ROS_INFO_STREAM("Publishing point cloud result from CNN.");
      cloud_pub.publish(cloud_cf);

      // now we have a pointcloud in camera frame
      // lets put it in world frame, so that we can find the lowest point in the z world direction
      PointCloudT::Ptr meshVerticesWorldFrame(new PointCloudT());
      pcl::transformPointCloud(*meshVerticesCameraFrame, *meshVerticesWorldFrame, transformEigen);

      // now we have a pointcloud in camera frame
      // lets put it in world frame, so that we can find the lowest point in the z world direction
      PointCloudT::Ptr partialCloudWorldFrame(new PointCloudT());
      pcl::transformPointCloud(*cloud, *partialCloudWorldFrame, transformEigen);

      Eigen::Vector4f centroid (0.f, 0.f, 0.f, 1.f);
      pcl::compute3DCentroid (*meshVerticesWorldFrame, centroid); centroid.w () = 1.f;
      ROS_INFO_STREAM("centroid.x(): " << centroid.x() << std::endl);
      ROS_INFO_STREAM("centroid.y(): " << centroid.y() << std::endl);
      ROS_INFO_STREAM("centroid.z(): " << centroid.z() << std::endl);
      // Eigen::Matrix4f world2Mesh = Eigen::Matrix4f::Identity ();
      // world2Mesh(0,3) = -centroid.x();
      // world2Mesh(1,3) = -centroid.y();
      // world2Mesh(2,3) = 0;
      // world2Mesh(3,3) = 1;
      // ROS_INFO_STREAM( "World2Mesh" << world2Mesh << std::endl);
      //
      // //from world frame to mesh frame.
      // PointCloudT::Ptr meshVerticesMeshFrame(new PointCloudT());
      // pcl::transformPointCloud(*meshVerticesWorldFrame, *meshVerticesMeshFrame, world2Mesh);
      // PointCloudT::Ptr partialCloudMeshFrame(new PointCloudT());
      // pcl::transformPointCloud(*partialCloudWorldFrame, *partialCloudMeshFrame, world2Mesh);
      //
      // //now we need to repackage the pointcloud as a mesh
      // for (int i =0 ; i < meshVerticesMeshFrame->size(); i++){
      //     geometry_msgs::Point geom_p_msg;
      //
      //     geom_p_msg.x = meshVerticesMeshFrame->at(i).x;
      //     geom_p_msg.y = meshVerticesMeshFrame->at(i).y;
      //     geom_p_msg.z = meshVerticesMeshFrame->at(i).z;
      //
      //     meshMsg.vertices.push_back(geom_p_msg);
      // }
      //
      //  meshMsg.triangles = result->mesh.triangles;
      //
      //  pcl::toROSMsg(*partialCloudMeshFrame, partialCloudMsg);
      //  ROS_INFO_STREAM("PARTIAL CLOUD SIZE: " << partialCloudMeshFrame->size() << std::endl);
      // now we have a mesh with the origin at the center of the base in the world frame of reference
      // lets get the transform from world to mesh center
       ROS_INFO_STREAM("OBJECT POSE IN FRAME: " << world_frame << std::endl);
       poseStampedMsg.header.frame_id = world_frame;
       poseStampedMsg.pose.orientation.w = 1.0;
       poseStampedMsg.pose.position.x = centroid.x();
       poseStampedMsg.pose.position.y = centroid.y();
       poseStampedMsg.pose.position.z = 0;
   }
}
