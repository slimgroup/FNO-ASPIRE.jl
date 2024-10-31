# Move contents of weights/DFNO_3D to weights/$1, then recreate weights/DFNO_3D
mv weights/DFNO_3D weights/"$1"
mkdir weights/DFNO_3D

# Move contents of plots/DFNO_3D to plots/$1, then recreate plots/DFNO_3D
mv plots/DFNO_3D plots/"$1"
mkdir plots/DFNO_3D

# Move specific contents from FNO/plots/DFNO_3D to plots/$1/DFNO_3D, then recreate original folder
mv FNO/plots/DFNO_3D/* plots/"$1"/DFNO_3D
