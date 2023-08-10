import { useState } from "react"
import { Button, Grid, Stack } from "@mui/material"
import { BatteryConfig } from "./BatteryConfig"
import { SolarConfig } from "./SolarConfig"

export const ConfigForm = () => {
  const [maxCRate, setMaxCRate] = useState(1)
  const [packEnergyCap, setPackEnergyCap] = useState(8e4)
  const [packMaxAh, setPackMaxAh] = useState(250)
  const [packMaxVoltage, setPackMaxVoltage] = useState(400)
  const [packVoltage, setPackVoltage] = useState(350)
  const [soh, setSoh] = useState(1)
  const [soc, setSoc] = useState(0.6)
  const [file, setFile] = useState<File | undefined>()
  const [startYear, setStartYear] = useState(2018)
  const [startMonth, setStartMonth] = useState(1)

  const submitConfigForm = async() => {
    const formData = new FormData()

    formData.append('max_c_rate', maxCRate.toString())
    formData.append('pack_energy_cap', packEnergyCap.toString())
    formData.append('pack_max_ah', packMaxAh.toString())
    formData.append('pack_max_voltage', packMaxVoltage.toString())
    formData.append('pack_voltage', packVoltage.toString())
    formData.append('SOH', soh.toString())
    formData.append('SOC', soc.toString())

    if (file) {
      formData.append('file', file)
    }
    formData.append('start_year', startYear.toString())
    formData.append('start_month', startMonth.toString())

    try {
      const response = await fetch('/run', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Request failed');
      }

      const responseData = await response.json();
      console.log('Server response:', responseData.message);
    } catch (error) {
      console.error('Error:', error);
    }
  }

  return (
    <form>
      <Stack spacing={0.5} maxWidth='md' m='auto'>
        <BatteryConfig 
          setMaxCRate={setMaxCRate}
          setPackEnergyCap={setPackEnergyCap}
          setPackMaxAh={setPackMaxAh}
          setPackMaxVoltage={setPackMaxVoltage}
          setPackVoltage={setPackVoltage}
          setSoh={setSoh}
          setSoc={setSoc}
          defaultExpanded
        />
        <SolarConfig
          setFile={setFile}
          setStartYear={setStartYear}
          setStartMonth={setStartMonth}
        />
        <Grid container>
          <Grid item xs={4}>
            <Button size='large' variant='contained' onClick={async() => { await submitConfigForm()} }>Simulate</Button>
          </Grid>
          <Grid item xs={8} />
        </Grid>
      </Stack>
    </form>
  )
}