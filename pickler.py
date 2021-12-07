import pickle

from processing.py import sentence_maps, drug_maps, output_map_filename


# write maps to .pkl
maps = (sentence_maps, drugs_maps)
print(maps[1][2])
pkl.dump(maps, open(output_map_filename, "wb"))

#test retrieving from .pkl
maps_temp = pkl.load(open(output_map_filename, "rb"))
print(maps_temp[1][2])
