import pickle
import imageio
import cv2
import numpy as np
from collections import deque

def predict_on_single_action(model ,video_path ,SEQUENCE_LENGTH ,IMAGE_HEIGHT ,IMAGE_WIDTH ,CLASSES_LIST):
    frame_deque = deque(maxlen=SEQUENCE_LENGTH)

    video_reader = cv2.VideoCapture(video_path)

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH) ,1)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES ,frame_counter * skip_frames_window)

        success ,frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame ,(IMAGE_HEIGHT ,IMAGE_WIDTH))

        normalized_frame = resized_frame / 255

        frame_deque.append(normalized_frame)

    video_reader.release()

    predicted_probs = model.predict(np.expand_dims(frame_deque ,axis=0))[0]

    predicted_label =  np.argmax(predicted_probs)

    predicted_class_name = CLASSES_LIST[predicted_label]

    return predicted_class_name , predicted_probs[predicted_label]
        


def predict_on_video(model ,video_path ,output_path ,SEQUENCE_LENGTH ,IMAGE_HEIGHT ,IMAGE_WIDTH ,CLASSES_LIST):
    video_reader = cv2.VideoCapture(video_path)
    
    video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    video_writer = imageio.get_writer(output_path + '.mp4', fps=video_reader.get(cv2.CAP_PROP_FPS))
    
    frame_deque = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while video_reader.isOpened():
        success, frame = video_reader.read()

        if not success:
            break

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frame_deque.append(normalized_frame)

        if len(frame_deque) == SEQUENCE_LENGTH:
            predicted_probs = model.predict(np.expand_dims(frame_deque, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use imageio to append the frame to the video
        video_writer.append_data(frame)

    video_reader.release()
    video_writer.close()

def predict_on_webcam(model, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST):
    video_capture = cv2.VideoCapture(0) 

    frame_deque = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class_name = ''

    while True:
        success, frame = video_capture.read()

        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frame_deque.append(normalized_frame)

        if len(frame_deque) == SEQUENCE_LENGTH:
            predicted_probs = model.predict(np.expand_dims(frame_deque, axis=0))[0]
            predicted_label = np.argmax(predicted_probs)
            predicted_class_name = CLASSES_LIST[predicted_label]

        cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    
def main():
    
    with open('convlstm_model.pickle' ,'rb') as file:
        model = pickle.load(file)
        
    video_path = 'test_video.mp4'
    SEQUENCE_LENGTH = 20
    IMAGE_HEIGHT ,IMAGE_WIDTH = 64 ,64
    CLASSES_LIST = ['Biking','PullUps','PushUps','Swing']
    
    # predicted_class , predicted_probablity  = predict_on_single_action(model ,video_path ,SEQUENCE_LENGTH ,IMAGE_HEIGHT ,IMAGE_WIDTH ,CLASSES_LIST)
    
    # print(f'predicted_class:{predicted_class} , predicted_probablity:{predicted_probablity}')
    
    predict_on_webcam(model, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST)

    
if __name__ == '__main__':
    main()