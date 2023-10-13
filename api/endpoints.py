from fastapi import APIRouter, HTTPException
from service.image_analyzer import ColorAnalyzer
from dto.image_data import ColorExtractionRequest, ColorExtractionResponse
import logging


#define the router for the app
router = APIRouter()

logging.basicConfig(
        format = '%(levelname)s:     %(asctime)s, %(module)s, %(processName)s, %(message)s',
        level = logging.INFO)

logger = logging.getLogger(__name__)

#define a post request for the /colors endpoint

@router.post(
        '/colors',                                  #endpoint name
        response_model = ColorExtractionResponse,   #data model
        tags = ['Colors Extraction']                #tags for documentation purpose
        )


#define an asynchronus function acception a 'ColorExtractionRequest' as request body
async def colors(input_data: ColorExtractionRequest):
    '''
    Analyze an image and return predominant colors.
    
    Parameters:
      - input_data[ColorExtractionRequest]: Request data including 'url_or_path' (str) and 'num_clusters' (int, optional).

    Returns:
      - ImageAnalysisResponse: Response data containing a list of predominant colors.

    Example Usage:
      - Send a POST request with JSON data containing the 'url_or_path' parameter to extract colors from an image.
    '''

    #log the request information
    logger.info(f'Analysis for image key: {input_data.url_or_path}.')
    logger.info(f'Requested colors: {input_data.num_clusters}.')

    #perform the color extraction

    try:

        #instantiate the ColorAnalyzer class for image processing
        color_json = ColorAnalyzer(
                input_data.url_or_path,
                input_data.num_clusters
            ).get_predominant_colors()

        logger.info(f'Analysis completed.')

        #return the predominant colors
        return {'predominant_colors':color_json}

    #if there is an error

    except Exception as e:

        #log the error message
        logger.error(f'Exception in image processing: {str(e)}.')

        #Raise an exception
        raise HTTPException(status_code = 500, detail = str(e))


