from django.http import JsonResponse
from django.shortcuts import render
from .ml_utils import RealEstateAnalyzer
import pandas as pd
import traceback
import json
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def safe_float(value, default):
    try:
        return float(value) if value and str(value).strip() else default
    except (ValueError, TypeError):
        return default

def safe_int(value, default):
    try:
        return int(value) if value and str(value).strip() else default
    except (ValueError, TypeError):
        return default

def home(request):
    return render(request, 'website.html')

def predict(request):
    # Initialize analyzer once at the start
    analyzer = RealEstateAnalyzer()
    if not analyzer.load_models():
        return JsonResponse({'success': False, 'error': 'Failed to load models'}, status=500)

    if request.method == 'POST':
        try:
            # Mapping dictionaries
            apartment_type_map = {1: 'secondary', 2: 'new'}
            renovation_map = {1: 'no', 2: 'cosmetic', 3: 'euro', 4: 'designer'}

            # Get raw values from POST
            raw_apartment_type = safe_int(request.POST.get('apartment_type'), 1)
            raw_renovation = safe_int(request.POST.get('renovation'), 2)

            # Map to string values
            apartment_type = apartment_type_map.get(raw_apartment_type, 'secondary')
            renovation = renovation_map.get(raw_renovation, 'cosmetic')

            # Get metro stations dictionary from POST
            metro_stations = {}
            try:
                metro_stations_raw = request.POST.get('metro_stations', '{}')
                metro_stations = json.loads(metro_stations_raw)
            except json.JSONDecodeError:
                print("Warning: Invalid metro_stations JSON, proceeding without metro filtering")

            # Build raw_data dictionary
            raw_data = {
                'minutes_to_metro': safe_float(request.POST.get('minutes_to_metro'), 5.0),
                'number_of_rooms': safe_int(request.POST.get('number_of_rooms'), 1),
                'area': safe_float(request.POST.get('area'), 42.0),
                'living_area': safe_float(request.POST.get('living_area'), 32.0),
                'kitchen_area': safe_float(request.POST.get('kitchen_area'), 10.0),
                'floor': safe_int(request.POST.get('floor'), 5),
                'number_of_floors': safe_int(request.POST.get('number_of_floors'), 9),
                'metro_lat': safe_float(request.POST.get('metro_lat'), None),
                'metro_lon': safe_float(request.POST.get('metro_lon'), None),
                'apartment_type': apartment_type,
                'renovation': renovation
            }

            # Validate required fields
            required_fields = [
                'minutes_to_metro', 'number_of_rooms', 'area', 'living_area', 
                'kitchen_area', 'floor', 'number_of_floors', 'metro_lat', 
                'metro_lon', 'apartment_type', 'renovation'
            ]
            
            for field in required_fields:
                if raw_data[field] is None:
                    return JsonResponse({
                        'success': False, 
                        'error': f'Missing or invalid field: {field}'
                    }, status=400)

            # Validate categorical values
            valid_apartment_types = ['secondary', 'new']
            valid_renovations = ['no', 'cosmetic', 'euro', 'designer']
            
            if raw_data['apartment_type'] not in valid_apartment_types:
                return JsonResponse({
                    'success': False, 
                    'error': f'Invalid apartment type: {raw_data["apartment_type"]}'
                }, status=400)
                
            if raw_data['renovation'] not in valid_renovations:
                return JsonResponse({
                    'success': False, 
                    'error': f'Invalid renovation: {raw_data["renovation"]}'
                }, status=400)

            # Prepare form_data with proper column names for ml_utils
            form_data = {
                'minutes to metro': raw_data['minutes_to_metro'],
                'number of rooms': raw_data['number_of_rooms'],
                'area': raw_data['area'],
                'living area': raw_data['living_area'],
                'kitchen area': raw_data['kitchen_area'],
                'floor': raw_data['floor'],
                'number of floors': raw_data['number_of_floors'],
                'metro_lat': raw_data['metro_lat'],
                'metro_lon': raw_data['metro_lon'],
                'apartment type': raw_data['apartment_type'],
                'renovation': raw_data['renovation']
            }

            # Get data type preference
            use_cleaned_models = request.POST.get('data-type', 'cleaned') == 'cleaned'
            
            # Make prediction
            prediction = analyzer.predict_price(form_data, use_cleaned_models=use_cleaned_models)

            # Compute seg2_predictions if not provided
            seg2_predictions = prediction.get('seg2_predictions', None)
            if not seg2_predictions:
                seg4_proba = prediction['seg4_predictions']['probabilities']
                seg2_proba = {
                    '0': seg4_proba.get(0, 0.0) + seg4_proba.get(1, 0.0),
                    '1': seg4_proba.get(2, 0.0) + seg4_proba.get(3, 0.0) + seg4_proba.get(4, 0.0)
                }
                seg2_pred = max(seg2_proba, key=seg2_proba.get) if seg2_proba else 0
                seg2_confidence = max(seg2_proba.values()) if seg2_proba else 0.0
                seg2_predictions = {
                    'segment': int(seg2_pred),
                    'probabilities': {str(k): float(v) for k, v in seg2_proba.items()}
                }

            # Use seg2 segment for general_clean
            chosen_segment = prediction.get('chosen_segment')
            if prediction.get('chosen_system') == 'general_clean' and seg2_predictions:
                chosen_segment = seg2_predictions['segment']

            # Find similar properties with metro filtering
            similar_properties = []
            try:
                similar_properties = analyzer.find_similar_properties(
                    form_data, 
                    top_n=10, 
                    min_matches=5,
                    metro_stations=metro_stations
                )
                # Convert numpy types to Python types for JSON serialization
                for prop in similar_properties:
                    for key, value in prop.items():
                        if isinstance(value, (np.floating, np.integer)):
                            prop[key] = float(value) if isinstance(value, np.floating) else int(value)
            except Exception as e:
                print(f"Error in find_similar_properties: {str(e)}")
                print(traceback.format_exc())
                similar_properties = []

            # Build response compatible with the frontend
            response_data = {
                'success': True,
                'ensemble_prediction': float(prediction.get('ensemble_prediction', 0)),
                'predicted_price_range': prediction.get('predicted_price_range', '0 - 0'),
                'lower_bound': float(prediction.get('lower_bound', 0)),
                'upper_bound': float(prediction.get('upper_bound', 0)),
                'midpoint': float(prediction.get('midpoint', 0)),
                'chosen_system': prediction.get('chosen_system', 'general_clean'),
                'chosen_segment': chosen_segment,
                'confidence': float(prediction.get('confidence', 0.0)),
                'prediction_details': {
                    'rf': float(prediction.get('prediction_details', {}).get('rf', 0)),
                    'xgb': float(prediction.get('prediction_details', {}).get('xgb', 0)),
                    'nn': float(prediction.get('prediction_details', {}).get('nn', 0))
                },
                'model_weights': {
                    'RF': float(prediction.get('model_weights', {}).get('RF', 0.0)),
                    'XGB': float(prediction.get('model_weights', {}).get('XGB', 0.0)),
                    'NN': float(prediction.get('model_weights', {}).get('NN', 0.0))
                },
                'segmentation_analysis': {
                    'quartile_system': {
                        'segment': int(prediction.get('seg4_predictions', {}).get('segment', 0)),
                        'probabilities': {str(k): float(v) for k, v in prediction.get('seg4_predictions', {}).get('probabilities', {}).items()}
                    },
                    'tertile_system': {
                        'segment': int(prediction.get('seg3_predictions', {}).get('segment', 0)),
                        'probabilities': {str(k): float(v) for k, v in prediction.get('seg3_predictions', {}).get('probabilities', {}).items()}
                    },
                    'biseg_system': {
                        'segment': seg2_predictions['segment'],
                        'probabilities': seg2_predictions['probabilities']
                    }
                },
                'explanations': {
                    'shap': {k: float(v) for k, v in prediction.get('explanations', {}).get('shap', {}).items()},
                    'lime': {k: float(v) for k, v in prediction.get('explanations', {}).get('lime', {}).items()}
                },
                'model_errors': {
                    'rf': float(prediction.get('model_errors', {}).get('rf', 1)),
                    'xgb': float(prediction.get('model_errors', {}).get('xgb', 1)),
                    'nn': float(prediction.get('model_errors', {}).get('nn', 1))
                },
                'similar_properties': similar_properties
            }

            print(f"Response data: {response_data}")
            return JsonResponse(response_data)

        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            print(traceback.format_exc())
            return JsonResponse({
                'success': False, 
                'error': f'Prediction error: {str(e)}'
            }, status=400)
    
    return render(request, 'website.html')