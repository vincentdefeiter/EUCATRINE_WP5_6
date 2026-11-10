#!/bin/ksh
# Author: Vincent de Feiter. 
# Contact: vincent.defeiter@wur.nl

# Version: 13 October, 2026
#-------------------------------------------------------

set -e

#---------------------------------------------------------
# E X E C U T A B L E  -  D O W N L O A D I N G  D A T A 
#---------------------------------------------------------

module load cdo
module load eclib
module load ecmwf-toolbox

#Set buffer
export MARS_READANY_BUFFER_SIZE=768896196

#Steps
step1=0
step2=23
step_delta=1

#Get variables
EXPVER=$1
param=$2
date1=$3
date2=$4
area=$5
latresol=${6}
lonresol=${7}
STREAM=${8}
DATAPATH=${9}

echo "Testing input parameters are set correctly..."
echo EXPVER $EXPVER
echo param $param
echo date1 $date1
echo date2 $date2
echo area $area
echo arealon1 $arealon1
echo arealon2 $arealon2
echo arealat1 $arealat1
echo arealat2 $arealat2
echo latresol $latresol
echo lonresol $lonresol
echo DATAPATH $DATAPATH
echo STREAM $STREAM

TYPE=fc
analysis_only=False

#---------------------------------------------------------
#        H E L P E R  -  S K I P  C O M P L E T E D
#---------------------------------------------------------

check_completed()
{
    completed_request="$1"

    if [[ -s "${file}.grib" && -s "${file}.nc" && -f "${file}.done" ]] &&
        cmp -s "$completed_request" "${file}.done"; then
        echo "Already completed: ${file}.nc -- skipping"
        exit 0
    fi

    #Remove any old marker before attempting this request.
    rm -f "${file}.done"
}

echo ">>> Retrieving data from mars ..."

#---------------------------------------------------------
#           G E T  G R I B  F I L E S  
#---------------------------------------------------------

if [[ $param == "228080/228081/228082" ]]; then

    echo $param,"DOWNLOADING SURFACE FLUXES OF CO2"

cat << EOF > marsjob_${EXPVER}_sfc_co2vars_${date1}
retrieve,
time=0,
date=${date1}/to/${date2},
stream=${STREAM},
repres=ll,
area=$area,
grid=${lonresol}/${latresol},
step=${step1}/to/${step2}/by/${step_delta},
levtype=sfc,
interpolation="--interpolation=grid-box-average",
expver=${EXPVER},
type=${TYPE},
class=rd,
param=${param},
target="${DATAPATH}/${EXPVER}_sfc_ll_co2vars_${date1}.grib"
EOF

file=${DATAPATH}/${EXPVER}_sfc_ll_co2vars_${date1}

check_completed "marsjob_${EXPVER}_sfc_co2vars_${date1}"

/usr/local/bin/mars  marsjob_${EXPVER}_sfc_co2vars_${date1}

elif [[ $param == "228078" ]]; then

    echo $param,"DOWNLOADING GPP CORRECTION AT STEP 0"

cat << EOF > marsjob_${EXPVER}_sfc_gpp_${date1}
retrieve,
time=0,
date=${date1}/to/${date2},
stream=${STREAM},
repres=ll,
area=$area,
grid=${lonresol}/${latresol},
step=0,
levtype=sfc,
interpolation="--interpolation=nearest-neighbour",
expver=${EXPVER},
type=${TYPE},
class=rd,
param=${param},
target="${DATAPATH}/${EXPVER}_sfc_ll_gpp_${date1}.grib"
EOF

file=${DATAPATH}/${EXPVER}_sfc_ll_gpp_${date1}

check_completed "marsjob_${EXPVER}_sfc_gpp_${date1}"

/usr/local/bin/mars marsjob_${EXPVER}_sfc_gpp_${date1}


else

cat << EOF > marsjob_${EXPVER}_sfc_${date1}
retrieve,
time=0,
date=${date1}/to/${date2},
stream=${STREAM},
repres=ll,
area=$area,
grid=${lonresol}/${latresol},
step=${step1}/to/${step2}/by/${step_delta},
levtype=sfc,
interpolation="--interpolation=nearest-neighbour",
expver=${EXPVER},
type=${TYPE},
class=rd,
param=${param},
target="${DATAPATH}/${EXPVER}_sfc_ll_${date1}.grib"
EOF
/usr/local/bin/mars  marsjob_${EXPVER}_sfc_${date1}

file=${DATAPATH}/${EXPVER}_sfc_ll_${date1}

check_completed "marsjob_${EXPVER}_sfc_${date1}"

/usr/local/bin/mars  marsjob_${EXPVER}_sfc_${date1}

fi

#---------------------------------------------------------
#        C O N V E R T  T O  N E T C D F 
#---------------------------------------------------------

#File
grib_to_netcdf -k 3 -d 4 -o "${file}.nc" "${file}.grib"

echo "Extract area and convert to netcdf"

#Mark complete only after successful retrieval and conversion.
if [[ ! -s "${file}.grib" || ! -s "${file}.nc" ]]; then
    echo "Missing or empty output -- not marking complete" >&2
    exit 1
fi

cp "$completed_request" "${file}.done"