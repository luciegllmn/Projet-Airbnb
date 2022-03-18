{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Méthode : MediaPipe pour la détection du sourire sur des vidéos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Packages \n",
    "import pandas as pd \n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "\n",
    "#Package Mediapipe\n",
    "import mediapipe as mp\n",
    "\n",
    "#Package pour importer des vidéos\n",
    "import cv2\n",
    "\n",
    "#Package pour transformer les résultats en liste \n",
    "from protobuf_to_dict import protobuf_to_dict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import de la vidéo en .webm\n",
    "cap = cv2.VideoCapture('/Users/lucieguillaumin/Documents/A1_Papier important/Détails/Ysance part of Devoteam/Intercontrat /VocaCoach/Vidéos/Video3.webm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import de la vidéo en .mp4\n",
    "cap = cv2.VideoCapture('/Users/lucieguillaumin/Documents/A1_Papier important/Détails/Ysance part of Devoteam/Intercontrat /VocaCoach/Vidéos/profil.mp4')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# For webcam input:\n",
    "with mp_holistic.Holistic(\n",
    "    static_image_mode=False,\n",
    "    min_detection_confidence=0.5,\n",
    "    min_tracking_confidence=0.5) as holistic:\n",
    "  while cap.isOpened():\n",
    "    success, image = cap.read()\n",
    "    if not success:\n",
    "      print(\"Ignoring empty camera frame.\")\n",
    "      # If loading a video, use 'break' instead of 'continue'.\n",
    "      continue #break pour pas afficher la vidéo\n",
    "\n",
    "    # To improve performance, optionally mark the image as not writeable to\n",
    "    # pass by reference.\n",
    "    image.flags.writeable = False\n",
    "    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\n",
    "    results = holistic.process(image)\n",
    "\n",
    "    # Draw landmark annotation on the image.\n",
    "    image.flags.writeable = True\n",
    "    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "    mp_drawing.draw_landmarks(\n",
    "        image,\n",
    "        results.face_landmarks,\n",
    "        mp_holistic.FACEMESH_CONTOURS,\n",
    "        landmark_drawing_spec=None,\n",
    "        connection_drawing_spec=mp_drawing_styles\n",
    "        .get_default_face_mesh_contours_style())\n",
    "    mp_drawing.draw_landmarks(\n",
    "        image,\n",
    "        results.pose_landmarks,\n",
    "        mp_holistic.POSE_CONNECTIONS,\n",
    "        landmark_drawing_spec=mp_drawing_styles\n",
    "        .get_default_pose_landmarks_style())\n",
    "    # Flip the image horizontally for a selfie-view display.\n",
    "    cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))\n",
    "    if cv2.waitKey(5) & 0xFF == 27:\n",
    "      break\n",
    "cap.release()\n",
    "#cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#results.pose_landmarks\n",
    "#results.segmentation_mask"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Résultats point de repère"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "res_pose = results.pose_landmarks\n",
    "\n",
    "#Conversion des landmarks en liste \n",
    "keypoints = protobuf_to_dict(results.pose_landmarks)\n",
    "\n",
    "#Conversion en dataset\n",
    "data_res_pose = pd.concat({k: pd.Series(v) for k, v in keypoints.items()}).reset_index()\n",
    "data_res_pose.columns = ['landmark', 'index','xyzvis']\n",
    "data_res_pose.head()\n",
    "\n",
    "#Transformation de la colonne xyzvis en un DataFrame\n",
    "dataset_pose = pd.DataFrame.from_dict(list(data_res_pose['xyzvis']))\n",
    "print(dataset_pose.shape)\n",
    "dataset_pose.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset_pose.loc[9:10,:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "type(res_pose)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "res_pose_world = results.pose_world_landmarks\n",
    "\n",
    "#Conversion des landmarks en liste \n",
    "keypoints = protobuf_to_dict(res_pose_world)\n",
    "\n",
    "#Conversion en dataset\n",
    "data_res_pose_world = pd.concat({k: pd.Series(v) for k, v in keypoints.items()}).reset_index()\n",
    "data_res_pose_world.columns = ['landmark', 'index','xyzvis']\n",
    "data_res_pose_world.head()\n",
    "\n",
    "#Transformation de la colonne xyzvis en un DataFrame\n",
    "dataset_pose_world = pd.DataFrame.from_dict(list(data_res_pose_world['xyzvis']))\n",
    "print(dataset_pose_world.shape)\n",
    "dataset_pose_world.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Résultats pour le visage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "res_face = results.face_landmarks\n",
    "\n",
    "#Conversion des landmarks en liste \n",
    "keypoints = protobuf_to_dict(res_face)\n",
    "\n",
    "#Conversion en dataset\n",
    "data_res_face = pd.concat({k: pd.Series(v) for k, v in keypoints.items()}).reset_index()\n",
    "data_res_face.columns = ['landmark', 'index','xyz']\n",
    "data_res_face.head()\n",
    "\n",
    "#Transformation de la colonne xyz en un DataFrame\n",
    "dataset_face = pd.DataFrame.from_dict(list(data_res_face['xyz']))\n",
    "print(dataset_face.shape)\n",
    "dataset_face.head()"
   ]
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
 "nbformat_minor": 4
}
