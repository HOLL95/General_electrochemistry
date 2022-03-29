<?php

class CPE {

  public $qmag; // The magnitude of the CPE, equivalent to Farads
  public $alpha; // The argument of the CPE, between 0 and 1
  public $r; // Series resistance value, in ohms
  public $step; // The basic time step, in seconds
  public $writeEvery; // How often to write to file
  public $ramp; // Voltage ramp rate, in volts per second
  public $tend; // The simulation end time
  public $currentGain = 1e6; // Multiply current by this when writing output files
  public $rampTime = 10; // How long the voltage waveform ramps in seconds
  public $dwellTime = 10; // How long the voltage waveform dwells between ramps in seconds
  public $amplitude = 0.1; // Amplitude of the voltage ramp in millivolts
  public $waveT1; // Waveform time point 1
  public $waveT2; // Waveform time point 2
  public $waveT3; // Waveform time point 3
  public $waveT4; // Waveform time point 4
  public $fileNameStub = 'cpe-euler';

  public $gamma;
  public $convu = array(); // The pre-calculated convolution function
  public $qstepu = array(); // The increments of charge, for each simulation step
  public $results = array(); // The results of the simulation

  public function __construct($qmag, $alpha) {
    $this->qmag = $qmag;
    $this->alpha = $alpha;
    $this->gamma = $this->gamma($alpha);
  }

  public function createConv() {
    $exp = $this->alpha - 1; // Exponent
    $tendSteps = $this->tend / $this->step; // Total number of steps
    // This loop sets up the convolution array $convu
    for ($tloop = 1; $tloop <= $tendSteps; $tloop++) {
      $t = $tloop * $this->step;
      $this->convu[$tloop] = pow($t, $exp);
    }
  }

  public function runSim() {
    $this->createConv();
    $q = 0; // The amount of electrical charge that has flowed
    $write = 0;
    $tendSteps = $this->tend / $this->step; // Total number of steps
    for ($tloop = 0; $tloop <= $tendSteps; $tloop++) {
      $t = $tloop * $this->step;
      // $vTotal = $t * $this->ramp;
      $vTotal = $this->voltWave($t);
      $vCpe = $this->calcVcpe($tloop);
      $i = ($vTotal - $vCpe) / $this->r;
      $qstep = $i * $this->step;
      $this->qstepu[$tloop] = $qstep;
      if ($write <= 1) {
        $result = array('t' => $t, 'vTotal' => $vTotal, 'vCpe' => $vCpe);
        $result['i'] = $i * $this->currentGain;
        $this->results[] = $result;
        $write = $this->writeEvery;
        $perComp = round($tloop / $tendSteps * 100);
        echo "\rSimulating: {$perComp}% complete...";
      } else {
        $write--;
      }
      $q = $q + $qstep;
    }
    echo "\rSimulating: Done                   \n";
  }

  public function voltWave($t) {
    // Returns the voltage in millivolts
    if (!$this->waveT4) {
      $this->waveT1 = $this->rampTime;
      $this->waveT2 = $this->rampTime + $this->dwellTime;
      $this->waveT3 = $this->rampTime * 2 + $this->dwellTime;
      $this->waveT4 = ($this->rampTime + $this->dwellTime) * 2;
    }
    $t2 = fmod($t, $this->waveT4);
    if ($t2 <= $this->waveT1) {
      $v = $t2 * $this->ramp;
    } else if ($t2 <= $this->waveT2) {
      $v = $this->amplitude;
    } else if ($t2 <= $this->waveT3) {
      $v = ($this->waveT3 - $t2) * $this->ramp;
    } else {
      $v = 0;
    }
    return $v;
  }

  public function calcVcpe($tloop) {
    $total = 0;
    for($uloop = 0; $uloop < $tloop; $uloop++) {
      $total += $this->convu[$tloop - $uloop] * $this->qstepu[$uloop];
    }
    return $total / $this->qmag / $this->gamma;
  }

  public function gamma($x) {
    // This code is copied from https://rosettacode.org/wiki/Gamma_function#Procedural
    $a = array(1.0, 0.5772156649015329, -0.6558780715202539,
               -0.04200263503409524, 0.16653861138229148,
               -0.04219773455554433, -0.009621971527876973,
               0.0072189432466631, -0.0011651675918590652,
               -0.00021524167411495098, 0.0001280502823881162,
               -2.013485478078824e-05, -1.25049348214267e-06,
               1.1330272319817e-06, -2.0563384169776e-07,
               6.11609510448e-09, 5.00200764447e-09,
               -1.18127457049e-09, 1.0434267117e-10,
               7.78226344e-12, -3.69680562e-12, 5.1003703e-13,
               -2.058326e-14, -5.34812e-15, 1.22678e-15,
               -1.1813e-16, 1.19e-18, 1.41e-18, -2.3e-19, 2e-20);
    $y = $x - 1.0;
    $counta = count($a);
    $sum = $a[$counta - 1];
    for ($n = $counta - 2; $n >= 0; $n--) {
      $sum = $sum * $y + $a[$n];
    }
    return 1.0 / $sum;
  }

  public function writeResults() {
    $fp1 = fopen("{$this->fileNameStub}.csv", 'w');
    $fp2 = fopen("{$this->fileNameStub}.idf", 'w');
    fwrite($fp1, "t,vTotal,vCpe,i\n");
    fwrite($fp2, "primary_data\n");
    fwrite($fp2, "3\n"); // The number of fields on each row
    fwrite($fp2, count($this->results)."\n"); // The number of rows
    foreach($this->results as $r) {
      fwrite($fp1, "{$r['t']},{$r['vTotal']},{$r['vCpe']}");
      fwrite($fp1, ",".($r['i'] * $this->currentGain)."\n");
      fwrite($fp2, "{$r['t']} {$r['i']} {$r['vTotal']}\n");
    }
    fclose($fp1);
    fclose($fp2);
  }

  public function toTarget($startTime, $endTime, $spiceCurrentGain) {
    $target = array();
    $startN = $this->time2N($startTime);
    $endN = $this->time2N($endTime);
    $minAmps = 99999;
    for ($n = $startN; $n <= $endN; $n++) {
      $minAmps = min($this->results[$n]['i'], $minAmps);
    }
    for ($n = $startN; $n <= $endN; $n++) {
      $r = $this->results[$n];
      $arr = array('voltage' => $r['vTotal']);
      $arr['amps'] = ($r['i'] - $minAmps) / $this->currentGain * $spiceCurrentGain;
      $time = round($r['t'] - $startTime, 6);
      $target[strval($time)] = $arr;
    }
    return $target;
  }

  public function time2N($time) {
    // This method gets the nearest n-value corresponding to a specified time
    // It uses binary search, for speed.
    // It's not possible to use an array, because the keys of an array must be either int or string
    $n1 = 0;
    $n2 = count($this->results);
    while (($n2 - $n1) > 1) {
      $midN = intval(($n1 + $n2) * 0.5);
      $midT = $this->results[$midN]['t'];
      // echo "n1 = $n1, n2 = $n2, midN = $midN, midT = $midT, time = $time\n";
      if ($midT == $time) {
        // echo "returning $midN\n";
        return $midN;
      } else if ($midT > $time) {
        $n2 = $midN;
      } else {
        $n1 = $midN;
      }
    }
    $abs1 = abs($this->results[$n1]['t'] - $time);
    $abs2 = abs($this->results[$n2]['t'] - $time);
    // echo "abs1 = $abs1, abs2 = $abs2\n";
    $midN = ($abs1 < $abs2) ? $n1 : $n2;
    // echo "returning $midN\n";
    return $midN;
  }

}
