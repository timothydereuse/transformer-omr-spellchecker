import h5py
from data_management.vocabulary import Vocabulary


dset = h5py.File("processed_datasets/supervised_omr_targets_bymeasure.h5", "a")
v = Vocabulary(load_from_file=r"processed_datasets\vocab_big.txt")

fann = dset["test"]["omr"]["sq_in_E-flat_major_Op277__Fanny_Hensel-tposed.None"]
finals = fann[0] == v.wtv["barline.final"]
bar_locs = finals.nonzero()

bps = [1985, 7551, 11498]
mvt1 = fann[:, 0 : bps[0]]
mvt2 = fann[:, bps[0] : bps[1]]
mvt3 = fann[:, bps[1] : bps[2]]
mvt4 = fann[:, bps[2] :]

dset.create_dataset(
    "test/omr/sq_in_E-flat_major_Op277__Fanny_Hensel-MVT1tposed.None", data=mvt1
)
dset.create_dataset(
    "test/omr/sq_in_E-flat_major_Op277__Fanny_Hensel-MVT2tposed.None", data=mvt2
)
dset.create_dataset(
    "test/omr/sq_in_E-flat_major_Op277__Fanny_Hensel-MVT3tposed.None", data=mvt3
)
dset.create_dataset(
    "test/omr/sq_in_E-flat_major_Op277__Fanny_Hensel-MVT4tposed.None", data=mvt4
)


asdf = dset["test"]["omr"]["sq_No2__Aleksandr_Borodin-tposed.None"]
finals = asdf[0] == v.wtv["barline.final"]
bar_locs = finals.nonzero()

bps = [8613, 17216, 23534]

mvt1 = asdf[:, 0 : bps[0]]
mvt2 = asdf[:, bps[0] : bps[1]]
mvt3 = asdf[:, bps[1] : bps[2]]

dset.create_dataset("test/omr/sq_No2__Aleksandr_Borodin-MVT1-tposed.None", data=mvt1)
dset.create_dataset("test/omr/sq_No2__Aleksandr_Borodin-MVT2-tposed.None", data=mvt2)
dset.create_dataset("test/omr/sq_No2__Aleksandr_Borodin-MVT3-tposed.None", data=mvt3)

schubert = "sq_In_D_Minor_D_810_Death_and_The_Maiden__Franz_Schubert-tposed.None"

asdf2 = dset["test"]["omr"][schubert]

finals = asdf2[0] == v.wtv["barline.final"]
bar_locs = finals.nonzero()

bps = [6372, 16056, 22154]

mvt1 = asdf2[:, 0 : bps[0]]
mvt2 = asdf2[:, bps[0] : bps[1]]
mvt3 = asdf2[:, bps[1] : bps[2]]
mvt4 = asdf2[:, bps[2] :]

dset.create_dataset(
    "test/omr/sq_In_D_Minor_D_810_Death_and_The_Maiden__Franz_Schubert-MVT1-tposed.None",
    data=mvt1,
)
dset.create_dataset(
    "test/omr/sq_In_D_Minor_D_810_Death_and_The_Maiden__Franz_Schubert-MVT2-tposed.None",
    data=mvt2,
)
dset.create_dataset(
    "test/omr/sq_In_D_Minor_D_810_Death_and_The_Maiden__Franz_Schubert-MVT3-tposed.None",
    data=mvt3,
)
dset.create_dataset(
    "test/omr/sq_In_D_Minor_D_810_Death_and_The_Maiden__Franz_Schubert-MVT4-tposed.None",
    data=mvt4,
)

asdf3 = dset["test"]["omr"]["sq_No1_Op41_No1__Robert_Schumann-tposed.None"]
finals = asdf3[0] == v.wtv["barline.final"]
bar_locs = finals.nonzero()

bps = [12238, 20075, 23710]
mvt1 = asdf3[:, 0 : bps[0]]
mvt2 = asdf3[:, bps[0] : bps[1]]
mvt3 = asdf3[:, bps[1] : bps[2]]
mvt4 = asdf3[:, bps[2] :]

dset.create_dataset(
    "test/omr/sq_No1_Op41_No1__Robert_Schumann-MVT1-tposed.None",
    data=mvt1,
)
dset.create_dataset(
    "test/omr/sq_No1_Op41_No1__Robert_Schumann-MVT2-tposed.None",
    data=mvt2,
)
dset.create_dataset(
    "test/omr/sq_No1_Op41_No1__Robert_Schumann-MVT3-tposed.None",
    data=mvt3,
)
dset.create_dataset(
    "test/omr/sq_No1_Op41_No1__Robert_Schumann-MVT4-tposed.None",
    data=mvt4,
)

dset.create_dataset(
    "test/omr/sq_in_E-flat_major_Op277__Fanny_Hensel-tposed.None",
    data=dset["train"]["omr"]["sq_in_E-flat_major_Op277__Fanny_Hensel-tposed.None"][:],
)

dset.create_dataset(
    "test/omr/sq_No1_Op27__Edvard_GriegIV_MVT1-tposed.None",
    data=dset["train"]["omr"]["sq_No1_Op27__Edvard_GriegIV_MVT1-tposed.None"][:],
)

asdf4 = dset["test"]["omr"]["sq_Teresa_Carreo-tposed.None"]
finals = asdf3[0] == v.wtv["barline.final"]
bar_locs = finals.nonzero()

bps = [5798, 10501, 19754]
mvt1 = asdf4[:, 0 : bps[0]]
mvt2 = asdf4[:, bps[0] : bps[1]]
mvt3 = asdf4[:, bps[1] : bps[2]]
mvt4 = asdf4[:, bps[2] :]

dset.create_dataset(
    "test/omr/sq_Teresa_Carreo-MVT1-tposed.None",
    data=mvt1,
)
dset.create_dataset(
    "test/omr/sq_Teresa_Carreo-MVT2-tposed.None",
    data=mvt2,
)
dset.create_dataset(
    "test/omr/sq_Teresa_Carreo-MVT3-tposed.None",
    data=mvt3,
)
dset.create_dataset(
    "test/omr/sq_Teresa_Carreo-MVT4-tposed.None",
    data=mvt4,
)
