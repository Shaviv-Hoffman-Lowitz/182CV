cd Tiny-ImageNet-C
for corruption in *
do
	cd "$corruption"
	for severity in *
	do
		cd "$severity"
        	rsync -a --include '*/' --exclude '*' . "/content/src"
        	for class in *
        	do
          		cd "$class"
          		for file in test* 
          		do
            			extension="${file##*.}"
            			filename="${file%.*}"
            			mv "$file" "/content/src/${class}/${filename}"-"${corruption}"-"${severity}"."${extension}"
          		done
          		cd ..
        	done
        	echo "Done with ${corruption}-${severity}"
		cd ..
	done
	cd ..
done
cd ..
