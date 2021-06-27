import photosearch.db
import glob
from photosearch.objectdetection import get_object_names
import photosearch.facedetection as f
from sklearn.cluster import DBSCAN

def create_tables():
    con = photosearch.db.connect()
    query = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_path TEXT NOT NULL,
    labels TEXT
);

CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding ARRAY NOT NULL,
    image_id INTEGER,
    cluster INTEGER,
    FOREIGN KEY (image_id) REFERENCES images (id)
);
"""
    con.executescript(query)



def load_images():
    con = photosearch.db.connect()
    cursor = con.cursor()

    for image_path in glob.glob('./pictures/*.jpg'):
        labels = get_object_names(image_path)
        cursor.execute("INSERT INTO images (image_path,labels) VALUES (?,?)", [image_path, ','.join(labels)])
        image_id = cursor.lastrowid
        faces = f.detect_faces(image_path)
        embeddings = f.get_face_embeddings(faces)
        for e in embeddings:
            cursor.execute("INSERT INTO faces(embedding, image_id) values (?, ?)", [e, image_id])

    cursor.close()
    con.commit()
    con.close()

def create_clusters():
    con = photosearch.db.connect()
    cursor = con.cursor()
    cursor.execute("SELECT * FROM faces")
    faces = cursor.fetchall()
    samples = [face[1] for face in faces]
    dbscan = DBSCAN(min_samples=2,eps=0.3, metric='cosine')
    clusters = dbscan.fit_predict(samples)
    for i, face in enumerate(faces):
        cluster_id = clusters[i].item() # to convert to a regular int from numpy
        if cluster_id != -1: # -1 is for noisy points not in a cluster
            cursor.execute("""UPDATE faces SET cluster = ? WHERE id = ?""", [cluster_id, face[0]])
    cursor.close()
    con.commit()
    con.close()

if __name__ == '__main__':
    create_tables()
    load_images()
    create_clusters()
