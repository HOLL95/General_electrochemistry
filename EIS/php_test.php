#!/usr/bin/php
<?php
$fp = fopen("cpe-euler.csv", "w");
fwrite($fp, "t,vTotal,vCpe,i\n");
fwrite($fp, "sec,V,V,A\n");
$qmag = 0.001;// The magnitude of the CPE, in pseudo-Farads
$alpha = 0.2;// The argument of the CPE, between 0 and 1
$r = 10;// Series resistance value, in ohms
$step = 0.001;// The basic time step, in seconds
$writeEvery = 10;// How often to write to file
$ramp = 0.01;// Voltage ramp rate, in volts per second
$tend = 10;// The simulation end time
$gamma = gamma($alpha);
echo $gamma;
$convu = array();// The convolution array
$qstepu = array();
$q = 0;// The amount of electrical charge that has flowed
$write = 0;
for ($tloop = 0;$tloop <=$tend /$step;$tloop++) {
  $t =$tloop *$step;
  $convu[$tloop] = ($tloop > 0) ? pow($t,$alpha - 1) : 0;
  $vTotal =$t *$ramp;
  $vCpe = cpe($alpha,$tloop,$t);
  //echo $vCpe, "\n";
  $i = ($vTotal -$vCpe) /$r;
  $qstepu[$tloop] =$i *$step;
  if ($write <= 1) {
    fwrite($fp, "$t,$vTotal,$vCpe,$i\n");
    $write =$writeEvery;
  } else {
    $write--;
  }
  $q =$q +$i *$step;
}
fclose($fp);
function cpe($alpha,$tloop,$t) {
  global$gamma,$step,$qmag,$qstepu,$convu;
  $total = 0;
  //echo "start","\n";
  for($uloop = 0;$uloop <$tloop;$uloop++) {
    $total +=$convu[$tloop -$uloop] *$qstepu[$uloop];
    //echo $qstepu[$uloop], " " ;
  }

  return$total /$qmag /$gamma;
}
function gamma($x) {
  // https://rosettacode.org/wiki/Gamma_function#Procedural
  $a = array(1.0, 0.5772156649015329, -0.6558780715202539,
  -0.04200263503409524, 0.16653861138229148,
  -0.04219773455554433, -0.009621971527876973,0.0072189432466631, -0.0011651675918590652,
  -0.00021524167411495098, 0.0001280502823881162,
  -2.013485478078824e-05, -1.25049348214267e-06,
  1.1330272319817e-06, -2.0563384169776e-07,
  6.11609510448e-09, 5.00200764447e-09,
  -1.18127457049e-09, 1.0434267117e-10,
  7.78226344e-12, -3.69680562e-12, 5.1003703e-13,
  -2.058326e-14, -5.34812e-15, 1.22678e-15,
  -1.1813e-16, 1.19e-18, 1.41e-18, -2.3e-19, 2e-20);
  $y =$x - 1.0;
  $counta = count($a);
  $sum =$a[$counta - 1];
  for ($n =$counta - 2;$n >= 0;$n--) {
    $sum =$sum *$y +$a[$n];
  }
  return 1.0 /$sum;
}
