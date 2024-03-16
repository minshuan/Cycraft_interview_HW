import tinydb
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="insert document in folder")
    parser.add_argument("--folder", type=str,default='news', help="folder name you want to insert")
    args = parser.parse_args()

    folder = args.folder
    db = tinydb.TinyVectorDB()
    num = 0
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            db.insert(content)
            num+=1

    db.build_index()
    print("insert %d document to tinydb"%num)
    db.save()