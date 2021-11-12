TARGET_DIR=pretrained_models/
# Check if all pretrained models already exists
if [ -f $TARGET_DIR/temporal_alignment_fold_1/model_epoch_90.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_2/model_epoch_30.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_3/model_epoch_29.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_4/model_epoch_74.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_5/model_epoch_83.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_6/model_epoch_66.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_7/model_epoch_84.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_8/model_epoch_91.pth ] &&
   [ -f $TARGET_DIR/temporal_alignment_fold_9/model_epoch_84.pth ] ; then
  echo 'Dataset already downloaded!'
  return
fi

CURRENT_DIR=${PWD}

#############################
# Creating folder           #
#############################
if [ ! -d $TARGET_DIR ]; then
  echo "Creating directory $TARGET_DIR"
  mkdir -p $TARGET_DIR/
fi

#############################
# Download from googledrive #
#############################
# Credits: https://gist.github.com/darencard/079246e43e3c4b97e373873c6c9a3798

# Check if dataset already exists
if [ ! -f $TARGET_DIR/temporal_pretrained_models.zip ]; then
  # googledrive link: https://drive.google.com/file/d/158t9PMDA6SbKNlExNVdw6r2UuqgyHf_A/view?usp=sharing
  gURL=158t9PMDA6SbKNlExNVdw6r2UuqgyHf_A

  ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')
  ggURL='https://drive.google.com/uc?export=download'
  curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
  getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
  cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
  echo -e "Downloading from "$gURL"...\n"
  cd $TARGET_DIR
  eval $cmd
fi

cd $CURRENT_DIR

if [ -f $TARGET_DIR/temporal_pretrained_models.zip ]; then
  #############################
  # Unzip                     #
  #############################
  echo 'Extracting file:'
  unzip $TARGET_DIR/temporal_pretrained_models.zip -d $TARGET_DIR
  #############################
  # Delete downloaded .zip    #
  #############################
  echo 'Removing the downloaded .zip file:'
  rm $TARGET_DIR/temporal_pretrained_models.zip
fi
