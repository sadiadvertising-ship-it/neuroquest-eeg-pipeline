import os
import time
from datetime import datetime
from .ganglion_stream import stream_eeg_to_csv
from ..config import DATA_DIR, CLASSES

def calibration_protocol():
    """Guía al usuario a través del protocolo de calibración para las 3 emociones."""
    
    print("--- PROTOCOLO DE CALIBRACIÓN DE EMOCIONES ---")
    print("Preparate para la sesión. Sigue las instrucciones.")
    print("Asegúrate de que tu Ganglion está conectada y encendida.")
    
    session_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    for round_num in range(3): # 3 repeticiones
        for emotion in CLASSES:
            print(f"\nPreparando para la emoción: **{emotion.upper()}**")
            print(f"Tendrás 1 minuto para sentir/mantener esta emoción.")
            input("Presiona ENTER para comenzar...")

            file_name = f"session_{session_id}_{emotion.lower()}_{round_num+1}.csv"
            output_path = os.path.join(DATA_DIR, 'raw', file_name)

            # Ensure the output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            stream_eeg_to_csv(duration_sec=60, output_file=output_path, label=emotion)
            
            print(f"\nBloque de {emotion.upper()} completado.")
            time.sleep(10) # Descanso entre bloques

    print("\n--- ¡CALIBRACIÓN COMPLETA! ---")
    print("Los datos se han guardado en la carpeta data/raw/.")
    print("Ahora puedes entrenar el modelo con el script train_and_eval.py.")

if __name__ == '__main__':
    calibration_protocol()