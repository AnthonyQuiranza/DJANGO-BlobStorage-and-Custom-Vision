from django.shortcuts import render
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import cv2
from PIL import Image, ImageDraw, ImageFont
import urllib
from azure.storage.blob import BlobServiceClient,BlobClient

connection_string = "<Blob Storage conection string>"
service = BlobServiceClient.from_connection_string(conn_str=connection_string)
def home(request):
    return render(request,"home.html")

def resultado(request):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.7
    fontColor              = (0,0,255)
    lineType               = 2
    name=request.GET["namefile"]
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": "<Prediction Key Aquí>"})
    predictor = CustomVisionPredictionClient("<Zona regional aqui>", credentials)
    blob = BlobClient.from_connection_string(conn_str=connection_string, container_name="images", blob_name=f"{name} training.png")
    url = request.GET["link"] 
    urllib.request.urlretrieve(url, "python.png")
    imagen=cv2.imread("python.png")
    height, width, channels = imagen.shape
    Resultado = predictor.detect_image_url("<Prediction Key>", "<Iteration>", url) 
    for prediction in Resultado.predictions:
        if prediction.probability > 0.4:
            bbox = prediction.bounding_box
            tag = prediction.tag_name
            probabilidad= int(prediction.probability * 100)
            result_image = cv2.rectangle(imagen, (int(bbox.left * width), int(bbox.top * height)), (int((bbox.left + bbox.width) * width), int((bbox.top + bbox.height) * height)), (0, 255, 0), 3)
            bottomLeftCornerOfText = (int(bbox.left*width),int(((bbox.top*height)+(bbox.height*height))))
            cv2.putText(result_image,str(probabilidad)+"% "+tag,
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
            cv2.imwrite('result.png', result_image)
    with open("result.png","rb") as data:
        blob.upload_blob(data)
    
    return render(request,"resultado.html",{"imagen":blob.url,"namefile":name})