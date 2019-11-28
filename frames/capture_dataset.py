# Program To Read video 
# and Extract Frames 
import cv2 
  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    cap = cv2.VideoCapture(0) 
  
    # Used as counter variable 
    count = 500
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, frame = cap.read() 
        
        cap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
  
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Saves the frames with frame-count 
        cv2.imwrite("dataset/paper/frame%d.jpg" % count, gray) 
  
        count += 1
        
        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
  
# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture("openCV.mp4") 