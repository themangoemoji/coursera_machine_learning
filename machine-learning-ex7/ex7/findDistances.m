function distance = findDistances (pts, centroid)
	distance = sum((pts-centroid).^2, 2);
end