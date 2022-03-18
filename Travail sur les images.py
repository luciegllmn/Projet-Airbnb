{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "15dfcd08-bbf7-43d8-9d8d-4e52d3b07895",
   "metadata": {},
   "source": [
    "# Application de la méthode MediaPipe aux images d'une vidéo  \n",
    "  \n",
    "Précedemment, on a capturé pour chaque vidéo une image toutes les 0.5 secondes : deux images par secondes.  \n",
    "\n",
    "On va donc d'abord appliqué l'algorithme pour une image, et ensuite appliquer une boucle pour récupérer pour chacune des images le résultat des points de la bouche.  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3e8c9f3d-f096-4c6f-9688-f8c8346b04ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Packages \n",
    "import pandas as pd \n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.image as mpimg\n",
    "\n",
    "#Package Mediapipe\n",
    "import mediapipe as mp\n",
    "\n",
    "#Package pour importer des vidéos\n",
    "import cv2\n",
    "\n",
    "#Lecture des images dans un dossier\n",
    "import os\n",
    "from os.path import isfile, join\n",
    "\n",
    "#Package pour transformer les résultats MediaPipe en liste \n",
    "from protobuf_to_dict import protobuf_to_dict\n",
    "\n",
    "#Chemins\n",
    "chemin_video = \"/Users/lucieguillaumin/Documents/A1_Papier important/Détails/Ysance part of Devoteam/Intercontrat /VocaCoach/Vidéos/\"\n",
    "chemin_img = \"/Users/lucieguillaumin/Documents/A1_Papier important/Détails/Ysance part of Devoteam/Intercontrat /VocaCoach/Vidéos/Par_images/\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "09e14d09-ac11-4262-8d86-8d0720cf752f",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Lecture des fichiers\n",
    "files = [f for f in os.listdir(chemin_img+ 'Face/') if isfile(join(chemin_img+ 'Face/', f))]\n",
    "#files"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10614426-1b38-4a65-8a46-7ac6d5480b17",
   "metadata": {},
   "source": [
    "## Application de MediaPipe à une image "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c53408fe-bc58-42e4-8056-a1a14faa57db",
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing = mp.solutions.drawing_utils\n",
    "mp_drawing_styles = mp.solutions.drawing_styles\n",
    "mp_holistic = mp.solutions.holistic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9bd43022-0e96-4f8e-b702-aa83007079c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "with mp_holistic.Holistic(\n",
    "    static_image_mode=True,\n",
    "    model_complexity=2,\n",
    "    enable_segmentation=True,\n",
    "    refine_face_landmarks=True) as holistic:\n",
    "    image = cv2.imread(chemin_img + 'Face/image1.jpg')\n",
    "    image_height, image_width, _ = image.shape\n",
    "    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "29bef8e2-a4c1-46e5-b2e1-ea68ce0b0d36",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(33, 4)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "      <th>z</th>\n",
       "      <th>visibility</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.567703</td>\n",
       "      <td>0.573513</td>\n",
       "      <td>-1.646631</td>\n",
       "      <td>0.998191</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.601432</td>\n",
       "      <td>0.498275</td>\n",
       "      <td>-1.573616</td>\n",
       "      <td>0.996980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.624694</td>\n",
       "      <td>0.495211</td>\n",
       "      <td>-1.574656</td>\n",
       "      <td>0.996127</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.643810</td>\n",
       "      <td>0.494251</td>\n",
       "      <td>-1.574379</td>\n",
       "      <td>0.996781</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.541673</td>\n",
       "      <td>0.510448</td>\n",
       "      <td>-1.568287</td>\n",
       "      <td>0.998308</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          x         y         z  visibility\n",
       "0  0.567703  0.573513 -1.646631    0.998191\n",
       "1  0.601432  0.498275 -1.573616    0.996980\n",
       "2  0.624694  0.495211 -1.574656    0.996127\n",
       "3  0.643810  0.494251 -1.574379    0.996781\n",
       "4  0.541673  0.510448 -1.568287    0.998308"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Conversion des landmarks en liste \n",
    "keypoints = protobuf_to_dict(results.pose_landmarks)\n",
    "\n",
    "#Conversion en dataset\n",
    "data_res_pose = pd.concat({k: pd.Series(v) for k, v in keypoints.items()}).reset_index()\n",
    "data_res_pose.columns = ['landmark', 'index','xyzvis']\n",
    "\n",
    "#Transformation de la colonne xyzvis en un DataFrame\n",
    "dataset_pose = pd.DataFrame.from_dict(list(data_res_pose['xyzvis']))\n",
    "print(dataset_pose.shape)\n",
    "dataset_pose.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bb2f489a-5d49-44bb-8666-28604e143e4b",
   "metadata": {},
   "source": [
    "Dans le DataFrame obtenu, on a 33 lignes qui correspondent au 33 points du visage.  \n",
    "Voici comment déterminer lesquelles correspondent à certain point : "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "a0380abd-1d11-4b76-8139-d1e3abeee04b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PoseLandmark.NOSE 0\n",
      "PoseLandmark.LEFT_EYE_INNER 1\n",
      "PoseLandmark.LEFT_EYE 2\n",
      "PoseLandmark.LEFT_EYE_OUTER 3\n",
      "PoseLandmark.RIGHT_EYE_INNER 4\n",
      "PoseLandmark.RIGHT_EYE 5\n",
      "PoseLandmark.RIGHT_EYE_OUTER 6\n",
      "PoseLandmark.LEFT_EAR 7\n",
      "PoseLandmark.RIGHT_EAR 8\n",
      "PoseLandmark.MOUTH_LEFT 9\n",
      "PoseLandmark.MOUTH_RIGHT 10\n",
      "PoseLandmark.LEFT_SHOULDER 11\n",
      "PoseLandmark.RIGHT_SHOULDER 12\n",
      "PoseLandmark.LEFT_ELBOW 13\n",
      "PoseLandmark.RIGHT_ELBOW 14\n",
      "PoseLandmark.LEFT_WRIST 15\n",
      "PoseLandmark.RIGHT_WRIST 16\n",
      "PoseLandmark.LEFT_PINKY 17\n",
      "PoseLandmark.RIGHT_PINKY 18\n",
      "PoseLandmark.LEFT_INDEX 19\n",
      "PoseLandmark.RIGHT_INDEX 20\n",
      "PoseLandmark.LEFT_THUMB 21\n",
      "PoseLandmark.RIGHT_THUMB 22\n",
      "PoseLandmark.LEFT_HIP 23\n",
      "PoseLandmark.RIGHT_HIP 24\n",
      "PoseLandmark.LEFT_KNEE 25\n",
      "PoseLandmark.RIGHT_KNEE 26\n",
      "PoseLandmark.LEFT_ANKLE 27\n",
      "PoseLandmark.RIGHT_ANKLE 28\n",
      "PoseLandmark.LEFT_HEEL 29\n",
      "PoseLandmark.RIGHT_HEEL 30\n",
      "PoseLandmark.LEFT_FOOT_INDEX 31\n",
      "PoseLandmark.RIGHT_FOOT_INDEX 32\n"
     ]
    }
   ],
   "source": [
    "for landmark in mp_holistic.PoseLandmark:\n",
    "    print(landmark, landmark.value)  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9ea60fac-ff0c-4cfc-b6cc-e0a3028c2182",
   "metadata": {},
   "source": [
    "## Application de MediaPipe pour toutes les images d'une vidéo\n",
    "On va donc ici appliquer une boucle dans l'algorithme."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cbceeed9-6f46-411a-bafd-6fdf2bdff732",
   "metadata": {},
   "outputs": [],
   "source": [
    "mp_drawing = mp.solutions.drawing_utils\n",
    "mp_drawing_styles = mp.solutions.drawing_styles\n",
    "mp_holistic = mp.solutions.holistic"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "59f87766-f680-4c96-b4d8-b590b1799455",
   "metadata": {},
   "outputs": [],
   "source": [
    "liste = []\n",
    "with mp_holistic.Holistic(\n",
    "        static_image_mode=True,\n",
    "        model_complexity=2,\n",
    "        enable_segmentation=True,\n",
    "        refine_face_landmarks=True) as holistic:\n",
    "            for i in range(len(files)):\n",
    "                filename = chemin_img + 'Face/' + files[i]\n",
    "                image = cv2.imread(filename)\n",
    "                image_height, image_width, _ = image.shape\n",
    "                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))\n",
    "                \n",
    "                if results.pose_landmarks:\n",
    "                    keypoints = protobuf_to_dict(results.pose_landmarks)\n",
    "                    data_res_pose = pd.concat({k: pd.Series(v) for k, v in keypoints.items()}).reset_index()\n",
    "                    data_res_pose.columns = ['landmark', 'index','xyzvis']\n",
    "                    dataset_pose = pd.DataFrame.from_dict(list(data_res_pose['xyzvis']))\n",
    "                    data_fin = dataset_pose.loc[9:10,:]\n",
    "                    liste.append([data_fin.loc[9,'x'], data_fin.loc[9,'y'], data_fin.loc[10,'x'], data_fin.loc[10,'y']])\n",
    "                "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "98ececba-ee69-4935-97fd-9d7cfd8a9c55",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Images dans la vidéo :  87\n",
      "Images pour lesquelles on obtient des résultats :  84\n"
     ]
    }
   ],
   "source": [
    "print('Images dans la vidéo : ', len(files))\n",
    "print('Images pour lesquelles on obtient des résultats : ', len(liste))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "67ebf101-fc9c-434f-9c28-4cf3fb9778cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gauche_x</th>\n",
       "      <th>Gauche_y</th>\n",
       "      <th>Droite_x</th>\n",
       "      <th>Droite_y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.469754</td>\n",
       "      <td>0.604013</td>\n",
       "      <td>0.435681</td>\n",
       "      <td>0.594387</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.565732</td>\n",
       "      <td>0.601171</td>\n",
       "      <td>0.524958</td>\n",
       "      <td>0.603276</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.457203</td>\n",
       "      <td>0.576065</td>\n",
       "      <td>0.425498</td>\n",
       "      <td>0.565234</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.511500</td>\n",
       "      <td>0.601793</td>\n",
       "      <td>0.502300</td>\n",
       "      <td>0.599173</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.663670</td>\n",
       "      <td>0.588388</td>\n",
       "      <td>0.611218</td>\n",
       "      <td>0.592042</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gauche_x  Gauche_y  Droite_x  Droite_y\n",
       "0  0.469754  0.604013  0.435681  0.594387\n",
       "1  0.565732  0.601171  0.524958  0.603276\n",
       "2  0.457203  0.576065  0.425498  0.565234\n",
       "3  0.511500  0.601793  0.502300  0.599173\n",
       "4  0.663670  0.588388  0.611218  0.592042"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Transformation de la liste en dataframe\n",
    "data = pd.DataFrame(liste, columns = ['Gauche_x','Gauche_y','Droite_x','Droite_y'])\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ff03444-44bf-4a57-9741-11e974214e04",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
