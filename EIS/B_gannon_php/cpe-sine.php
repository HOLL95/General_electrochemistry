#!/usr/bin/php
<?php
require_once('/home/wjg/Expts/classes/cpe.sine.class.php');

$expt = '../Expt-2022021101';
chdir($expt);
$fileNameStub = 'cpe-sine';

$qmag = 0.1; // The magnitude of the CPE, equivalent to Farads
$alpha = 0.8; // The argument of the CPE, between 0 and 1

$fp1 = fopen("{$fileNameStub}-z.csv", 'w');
fwrite($fp1, "freq,zMag,zPhi,zReal,zImag\n");
for ($logPN = 20; $logPN >= -20; $logPN--) {
  $cpeObj = new CPESine($qmag, $alpha);
  $cpeObj->fileNameStub = $fileNameStub;
  $cpeObj->currentGain = 1;
  $cpeObj->r = 1; // Series resistance value, in ohms
  $cpeObj->amplitude = 1; // Sine wave amplitude in volts (I think)
  $cpeObj->period = pow(10, $logPN / 10); // Since wave period in seconds
  $cpeObj->step = $cpeObj->period / 100; // The basic time step, in seconds
  $cpeObj->tend = $cpeObj->period * 5; // The simulation end time
  $cpeObj->runSim();
  $ret = $cpeObj->calcZ();
  fwrite($fp1, "{$ret['freq']},{$ret['zMag']},{$ret['zPhi']},{$ret['zReal']},{$ret['zImag']}\n");
  unset($cpeObj);
}
fclose($fp1);
