data/small:
	mkdir -p data/small

data/small/vehicles_smallset.zip: data/small
	wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles_smallset.zip -O data/small/vehicles_smallset.zip
	unzip data/small/vehicles_smallset.zip -d data/small

data/small/non-vehicles_smallset.zip: data/small
	wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles_smallset.zip -O data/small/non-vehicles_smallset.zip
	unzip data/small/non-vehicles_smallset.zip -d data/small

small-data: data/small/vehicles_smallset.zip data/small/non-vehicles_smallset.zip
	rm -rf data/small/__MAC*

data/full:
	mkdir -p data/full

data/full/vehicles.zip: data/full
	wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip -O data/full/vehicles.zip
	unzip data/full/vehicles.zip -d data/full

data/full/non-vehicles.zip: data/full
	wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip -O data/full/non-vehicles.zip
	unzip data/full/non-vehicles.zip -d data/full

full-data: data/full/vehicles.zip data/full/non-vehicles.zip
	rm -rf data/full/__MAC*