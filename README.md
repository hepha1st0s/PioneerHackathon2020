
# **AVATAR Video Conferencing**
## Idea
Video Conferencing can have a significant lag if you have a bad connection. To get the best out of video conferencing, even when connection speed is low, we are trying a new way to reduce the bits and bytes transferred by extracting the important information from the picture and mapping it onto an avatar.

Therefore the emotions are extracted in real time from the video stream and transmitted to the other clients - which match them onto the avatar.

This allows for shorter latency and better quality.

In the future, the avatar could be so realistic to deliver the impression that you experience a real video, while in fact just seeing your partners emotions and movements mapped onto a life realistic avatar.

## Implementation during Pioneer Hackathon
For the Hackathon we have decided to go for a quick solution to demo the extraction of information from an image including a face. Therefore we have implemented an algorithm that , utilizing python and OpenCV, can extract the face form an image (or camera feed). From the extracted face the algorithm evaluates wether the person in the image has an open mouth (showing the teeth) or the mouth is closed. The result is depicted directly in the image.

## Implementation Details
The solution can work with either test images or a live video feed. The algorithm extracts a face from the image data using a haar cascade (from [Github](https://github.com/opencv/opencv/tree/master/data/haarcascades)). If a face was found the algorithm extracts the mouth of the face in the same manner (also from Github but a [different Repository](https://github.com/peterbraden/node-opencv/tree/master/data)). After having extracted the mouth, the edges are calculated using a Canny Edge Detector. This will result in a black and white image representing the edges of the image in the mouth region. Based on this information, the algorithm decides whether the person in the image has an open mouth by simple thresholding on the number of white pixels (=edges) in the image. This can be done as teeth introduce a lot of additional hard edges in the image due to the interdental spaces. The edges of the image are displayed on top left of the result. The result is added to the image as a string next to it.


## Dependencies
[Python](https://www.python.org/)
[OpenCV for Python](https://opencv.org/)

## Demo Results
To demo the result we have decided to go for a dataset representing [Japanese Female Facial Expressions from 1989](https://zenodo.org/record/3451524#.XpLwZ1MzYkg). We have decided for this dataset as it was quick to download and free to use in publications when cited (so we should be safe now...).
