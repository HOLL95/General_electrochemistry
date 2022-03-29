<?php

require_once('cpe.class.php');

class CPESine extends CPE {

  public function voltWave($t) {
    // Returns the voltage in millivolts
    $rads = 2 * pi() * ($t / $this->period);
    $v = 0.5 * $this->amplitude * sin($rads);
    return $v;
  }

  public function calcZ() {
    $ret = array();
    $ret['freq'] = 1 / $this->period;
    $ret['tZero'] = $this->tend - $this->period;
    $found = False;
    $n = count($this->results) - 2;
    $i2 = $this->results[$n + 1]['i'];
    $tCrossN = -999; // Erroneous value
    while (!$found && ($n > 0)) {
      $i1 = $this->results[$n]['i'];
      if (!$found && ($i2 > 0) && ($i1 <= 0)) {
        // Found a negative crossing as we go backwards
        $ret['tCross'] = $this->results[$n]['t'];
        $tCrossN = $n;
        $found = True;
      }
      $positive = ($this->results[$n]['i'] > 0);
      $i2 = $i1;
      $n--;
    }
    if ($tCrossN > 0) {
      $periodN = $this->period / $this->step;
      $lastWave = array_slice($this->results, $tCrossN - $periodN, $periodN);
      foreach($lastWave as $r) {
        $iArr[] = $r['i'];
      }
      $ret['iMax'] = max($iArr);
      $ret['iMin'] = min($iArr);
      $ret['iMag'] = $ret['iMax'] - $ret['iMin'];
      $ret['zMag'] = $this->amplitude / $ret['iMag'];
      $ret['zPhi'] = 2 * pi() * ($ret['tCross'] - $ret['tZero']) / $this->period;
      $ret['zReal'] = $ret['zMag'] * cos($ret['zPhi']);
      $ret['zImag'] = $ret['zMag'] * sin($ret['zPhi']);
    }
    return $ret;
  }

}
