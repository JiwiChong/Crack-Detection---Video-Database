# Crack-Detection---Video-Database

Crack segmentation is not only essential to detect the extent of size of cracks in structures. It can 
also serve as a guidance or appliance to virtual or real worlds when it comes to crack inspection.
In this case, a Semantic Segmentation task was undertaken with real cracks across The University of New
Mexico. The detection and segmentation was carried out by Yolo-V7. It receives the data, including the
name of the image or video from an SQL database that is stored within phpMyAdmin. The output image or video
is then transferred to a different database. 

<div align="center">
<img src="https://github.com/user-attachments/assets/997d81f3-4bce-42b2-8396-85a5076d4774" width=90% height=85%>
</div><br />

Once the cracks were segmented, a script to segment cracks and extract the crack pixels as (X, Y)-coordinates 
was built to process both images and videos. 

Images with segmented cracks that were also detected with bounding boxes:

<div align="center">
<img src="https://github.com/user-attachments/assets/ca6c8c68-520f-43fb-a824-1103f0762ab1" width=90% height=85%>
</div><br />

Videos with crack pixels as (X, Y) coordinates:

<div align="center">
<img src="https://github.com/user-attachments/assets/50859104-000f-4c9b-ba51-0f34370ad6d1" width=90% height=85%>
</div><br />

The purpose of the script was to deploy the model and coordinates on a Microsoft HoloLens 2 Headset. Such work 
is promising for the Civil Engineers and Building inspectors as they would be able to inspect cracks in real 
construction sites without the need to carry a Server. Hence, this work offers convenience in terms of time 
saving and actual tool to identify cracks that would need to be addressed. 

<div align="left">
  <video src="https://github.com/user-attachments/assets/7160fcb4-c2e7-4887-8393-8bb963002194" width="200", height="200" />
</div>




