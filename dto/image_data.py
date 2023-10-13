#explore the data model for requests and responses

from pydantic import BaseModel
from typing import List


class Color(BaseModel):
    #color representation as RGB values

    R: int
    G: int
    B: int

class ColorInfo(BaseModel):
    #info abt color, rgb and percentage of pixels across the image.

    color: Color
    percentage: str

class ColorExtractionRequest(BaseModel):
    #colors extraction request.

    url_or_path: str
    num_clusters: int = 4

class ColorExtractionResponse(BaseModel):
    #response from an image analysis request

    predominant_colors: List[ColorInfo]

#accepts an input url or path as well as desired number of clusters/predominant colors
#returns a list of json objects made of RGB values and percentage of pixels in the img belonging to that cluster

#defining pydantic classes increases readability and maintance and also simplifies the generation of api documentation leveraging the api framework

