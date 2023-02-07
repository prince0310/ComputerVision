# Rep Count for Knee Bend Using MediaPipe Pose Model

### This project uses the MediaPipe Pose Model to detect and track the knee joint in a video stream, and then counts the number of times the knee is bent. The project is designed to work in real-time, making it useful for fitness and rehabilitation applications.

## Getting Started

####  Install the MediaPipe SDK by following the instructions on the MediaPipe website.
#### Clone or download this repository to your local machine.
``` git clone https://github.com/prince0310/kneeBend ```
#### Run the main.py script in the root directory of the project. This will start the video stream and begin counting knee bends.
``` python3 main.py --input filepath ```

#### Input 
> The project takes in a live video stream from the device's camera or video file.
#### Output
> The output of the project is the number of times the knee is bent, displayed in real-time on the screen. and also after quit output video will be saved in root directory
#### Customization

> The MediaPipe framework can be used with different models to detect and track multiple joints in the body. If you would like to track a different joint, you can modify the main.py script to use a different model instead. Additionally, you can change the threshold for what constitutes a "bend" to suit your needs. Also I have handle the fluctuation or dummy frame in video which can be avoide if video is smoot.

#### Limitations

> The performance of the model used in this project may vary depending on the quality of the input video, lighting conditions and the unique characteristics of the subject being tracked.

#### Acknowledgements

This project uses the MediaPipe framework, which was developed by the Google MediaPipe team. The framework is open-source and available on [MediaPipe github](https://github.com/google/mediapipe)

#### Output Video <br>



https://user-images.githubusercontent.com/85225054/213751977-8d63f6d6-e4f9-4415-abb4-bf32c78aae76.mp4













