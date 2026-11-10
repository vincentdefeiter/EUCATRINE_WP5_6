#!/bin/ksh
#SBATCH --job-name=A25SFC
#SBATCH --qos=el
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --output=mars-download.%j.log
#SBATCH --error=mars-download.%j.err
#SBATCH --mail-user=vincent.defeiter@wur.nl
#SBATCH --mail-type=BEGIN,END,FAIL

set -e

#-------------------------------------------------------
#            F I L E  S E T T I N G S
#-------------------------------------------------------
# Author: Vincent de Feiter. 
# Contact: vincent.defeiter@wur.nl

# Version: 13 October, 2026

#-------------------------------------------------------
#       D O W N L O A D   S E T T I N G S 
#-------------------------------------------------------

module load nco

#Compress to TAR files
tarfiles=false

#Location
sitename="Amazon"

#Resolution
EXPVER=itr2
resolinkm="25km"
STREAM=LWDA
latresol=0.18
lonresol=0.18

#EXPVER=iwpf 
#resolinkm="09km"
#STREAM=LWDA
# latresol=0.06
# lonresol=0.06

# EXPVER=j0xh
# resolinkm="05km" #4.4 km
# STREAM=LWDA
# latresol=0.03
# lonresol=0.03

#Datapath
DATAPATH=/scratch/ecme2192/${sitename}/${EXPVER}/SFC

if [[ ! -d $DATAPATH ]]; then
    mkdir -p $DATAPATH
fi    

#Area
area="15/-90/-60/-20"  #N/W/S/E

#Date range (full days)
yyyymmdd1="20220808"   #start
yyyymmdd2="20220820"   #end

#Variables
#Parameters that can be retrieved with shortname/paramID
param="159/228023/143/182/129/228046/188/172/142/66/67/186/187/228235/228236/235/147/175/177/176/134/173/146/169/130/164/228"
#"228080/228081/228082" for CO2 variables

#-------------------------------------------------------
#                   R E T R I E V A L 
#-------------------------------------------------------

d=$yyyymmdd1

#Loop over the dates
while [[ $d -le $yyyymmdd2 ]]; do

    echo Retrieving $d $param
    srun -c 4 --mem=15GB ksh run_extract_surface.sh $EXPVER $param $d $d $area $arealon1 $arealon2 $arealat1 $arealat2 $latresol $lonresol $STREAM $DATAPATH

    d=$(date -d"$d + 1 day" +"%Y%m%d")
done
