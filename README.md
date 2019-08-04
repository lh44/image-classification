The label to id Mapping

```
0 : airplane
1 : car
2 : cat
3 : dog
4 : flower
5 : fruit
6 : motorbike
7 : person
```

Running the app
1. Build the docker image
   
   >  docker build -t object-classification .

2. Run the app

   > docker run  object-classification

   The console output displays the results of the prediction on the images from random_images directory as 

```
image_1.jpg : 0
image_6.jpeg : 5
image_5.jpeg : 0
image_2.jpg : 1
image_8.jpg : 4
image_7.jpg : 6
image_3.jpeg : 2
image_4.jpg : 2
```