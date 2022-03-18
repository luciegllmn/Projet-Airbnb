{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "529f5518-c7ab-4175-aee3-abb92d9eb68e",
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
   "execution_count": null,
   "id": "7fb4b182-20da-4ab3-bb59-b00fa429c173",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import de la vidéo en .webm\n",
    "vidcap = cv2.VideoCapture(chemin_video + 'Video5.webm')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ba51350-a1ec-49b3-bb7c-2cf65e7880aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import de la vidéo en .mp4\n",
    "vidcap = cv2.VideoCapture(chemin_video + 'proche_lunettes.mp4')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23da14d8-c4e4-4b15-bfed-523039282c88",
   "metadata": {},
   "source": [
    "## Travail de la vidéo"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5bb4eee1-a0f5-4e56-83db-bbb196db11c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Fonction qui capture les images de la vidéo  \n",
    "def getFrame(sec):\n",
    "    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)\n",
    "    hasFrames,image = vidcap.read()\n",
    "    if hasFrames:\n",
    "        cv2.imwrite(\"image\"+str(count)+\".jpg\", image)     #On enregistre les photos dans le dossier\n",
    "    return hasFrames\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "804a208b-cb41-4987-86c7-8073c1292982",
   "metadata": {},
   "outputs": [],
   "source": [
    "sec = 0\n",
    "frameRate = 0.5 #Pour capturer des images toutes les 0.5 secondes\n",
    "count=1\n",
    "success = getFrame(sec)\n",
    "while success:\n",
    "    count = count + 1\n",
    "    sec = sec + frameRate\n",
    "    sec = round(sec, 2)\n",
    "    success = getFrame(sec)"
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
 "nbformat_minor": 5
}
