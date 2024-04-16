"""def makeTextFile(name, content):
    file = open(f"../youtube-transcript-summarizer-frontend/src/transcripts/{name}.txt","w",encoding="utf-8")
    
    file.write(f"{name} Transcript:\n")
    file.write(content)
    file.close()"""
import os

def makeTextFile(name, english_summary):
    directory = os.path.abspath("../youtube-transcript-summarizer-frontend/src/transcripts/")
    file_path = os.path.join(directory, f"{name}.txt")

    try:
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(english_summary)
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False