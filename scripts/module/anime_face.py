import cv2
import os

image_exts = [".png",".jpg",".jpeg"]

class AnimeFaceDetector:
    
    def __init__(self,padding,detection_output=False):
        # processing arguments
        self.padding=padding
        self.detection_output=detection_output # debug option
        
        extension_dir = os.path.dirname(__file__) # このファイルのディレクトリを参照する。
        # loading cascade
        cascade_file = f"lbpcascade_animeface.xml"
        cascade_path = os.path.join(extension_dir,cascade_file)
        assert os.path.isfile(cascade_path), f"{cascade_path} is not found!"
        self.cascade = cv2.CascadeClassifier(cascade_path)
    
    def __crop(self,cv_img,face_area,padding):
        max_width = cv_img.shape[0]
        max_height = cv_img.shape[1]
        x, y, w, h = face_area
        bottom,top,right,left = y+h+padding+1,y-padding,x+w+padding+1,x-padding
        
        bottom = min(bottom,max_height)
        top = max(top,0)
        right = min(right,max_width)
        left = max(left,0)
        crop_area = left,top,right,bottom
        _crop_img = cv_img[top:bottom,left:right]
        return _crop_img,crop_area
    
    def detect(self,image_path,output_directory,debug_output_directory):
        filename,fileext = os.path.splitext(os.path.basename(image_path))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
        if self.detection_output:
            detect_image = image.copy()
        # each face
        for i,face_area in enumerate(faces):
            crop_img,crop_area = self.__crop(image,face_area,self.padding)
            output_path = os.path.join(output_directory,f"{filename}-{i:02}{fileext}")
            cv2.imwrite(output_path, crop_img)
            if self.detection_output:
                x,y,bottom_x,left_y = crop_area
                cv2.rectangle(detect_image, (x, y), (bottom_x, left_y), (0, 0, 255), 2)
                
        if self.detection_output:
            output_path = os.path.join(debug_output_directory,f"{filename}-rect{fileext}")
            cv2.imwrite(output_path, detect_image)

        return len(faces) == 0
    

    
def detect(input_directory,output_directory,debug_output_directory,padding,detection_output):
    if debug_output_directory == None or debug_output_directory == "":
        debug_output_directory = output_directory
    afd = AnimeFaceDetector(padding,detection_output=detection_output)
    os.makedirs(output_directory,exist_ok=True)
    image_path_list = [os.path.join(input_directory,f) for f in os.listdir(input_directory) if any([f.endswith(ext) for ext in image_exts])]
    undetect_images = []
    for image_path in image_path_list:
        _detect = afd.detect(image_path,output_directory,debug_output_directory)
        if not _detect:
            undetect_images.append(image_path)
        
    results_text = "Face is not dound in imgaes:\n" + "\n".join(undetect_images)
    return results_text

