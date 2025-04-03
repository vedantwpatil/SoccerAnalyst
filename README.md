# SoccerAnalyst

Created a soccer soccer analyst which generates statistics from live data. This is done using computer vision models using yolo utilizing the pytorch framework for machine learning. The dataset was editing from the roboflow dataset

Currently I am working on improving the accuracy of the model specifically the ball tracking aspect of it. One of the biggest challenges in this is that the ball can move at multiple different speeds and the ball will look vastly different depending on the speed of the ball and the quality of the camera. This makes it really challenging to replicate in the data and have the model track the ball while it is speeding across the field. Additionally the model sometimes loses track of the ball in the air and struggles with tracking it against the different background
