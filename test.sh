# 공통 설정
GPU=0

# G3DNet18 설정
GBS=64
GOPT=Adam
GLR=0.001
LRS=0
L2=0
GNP=516

#RepSurf 설정
RBS=64
ROPT=Adam
RLR=0.001
DR=0.0001
DS=10
EP=100
RNP=1024


CUDA_VISIBLE_DEVICES=${GPU} python evaluate.py \
    --batch_size ${GBS} \
    --optimizer ${GOPT} \
    --learning_rate ${GLR} \
    --learning_rate_step ${LRS} \
    --l2 ${L2} \
    --num_points ${GNP} \


CUDA_VISIBLE_DEVICES=${GPU} python RepSurf/test_cls_sanlim.py \
    --batch_size ${RBS} \
    --optimizer ${ROPT} \
    --epoch ${EP} \
    --learning_rate ${RLR} \
    --decay_rate ${DR} \
    --decay_step ${DS} \
    --num_point ${RNP} \

CUDA_VISIBLE_DEVICES=${GPU} python ensemble.py \
    --g3dnet SurfG3D18_${GOPT}_np${GNP}_bs${GBS}_lr${GLR}_lrs${LRS}_wd${L2} \
    --repsurf RepSurf_${ROPT}_np${RNP}_bs${RBS}_lr${RLR}_dr${DR}_ds${DS} \