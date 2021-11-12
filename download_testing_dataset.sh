TARGET_DIR=dataset/testing_ds
# Check if dataset already exists
if [ -f $TARGET_DIR/temporal_align_val_test_lmdb/data.mdb ]; then
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
if [ ! -f $TARGET_DIR/temporal_align_val_test_lmdb.zip ]; then
  # googledrive link: https://drive.google.com/file/d/1QSKCm045K6oHi9w_gqFkGRUMsjHyuFko/view?usp=sharing
  gURL=1QSKCm045K6oHi9w_gqFkGRUMsjHyuFko

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

if [ -f $TARGET_DIR/temporal_align_val_test_lmdb.zip ]; then
  #############################
  # Unzip                     #
  #############################
  echo 'Extracting file:'
  unzip $TARGET_DIR/temporal_align_val_test_lmdb.zip -d $TARGET_DIR
  #############################
  # Delete downloaded .zip    #
  #############################
  echo 'Removing the downloaded .zip file:'
  rm $TARGET_DIR/temporal_align_val_test_lmdb.zip
fi
