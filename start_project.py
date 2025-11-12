from utils import process_audio_project

if __name__ == "__main__":
    project_root = "./projects"
    audio_info = {
        "clip_path": "audio/2023-12-03--10-15-10--Rec-C.sel.26.AA.wav",
        "nfft": 2048,
        "grayscale": True
    }

    result = process_audio_project(project_root, audio_info)

    print("\nSaved project data:")
    for k, v in result.items():
        print(f"  {k}: {v}")
