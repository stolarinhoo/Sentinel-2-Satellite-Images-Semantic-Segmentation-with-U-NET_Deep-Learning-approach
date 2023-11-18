# Constants
TRAIN_SIZE = 0.9
BATCH_SIZE = 32
INPUT_SHAPE = (256, 256, 12)  # 12 bands for input images
LABEL_SHAPE = (256, 256, 1)

FILTERS = 32
N_CLASSES = 9
EPOCHS = 60

# Classes
LAND_COVER_CLASSES = {
            0: "NO DATA",
            1: "Agricultural areas",
            2: "Forest and semi naturals",
            3: "Wetlands",
            4: "Water bodies",
            5: "Urban fabric",
            6: "Industrial, commercial and transport units",
            7: "Mine, dump and construction sites",
            8: "Artificial, non-agricultural vegetated areas"
        }
