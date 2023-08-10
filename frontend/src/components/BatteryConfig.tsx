import { Dispatch, SetStateAction } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Box, MenuItem, TextField, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'

interface BatteryConfigProps {
  setMaxCRate: Dispatch<SetStateAction<number>>;
  setPackEnergyCap: Dispatch<SetStateAction<number>>;
  setPackMaxAh: Dispatch<SetStateAction<number>>;
  setPackMaxVoltage: Dispatch<SetStateAction<number>>;
  setPackVoltage: Dispatch<SetStateAction<number>>;
  setSoh: Dispatch<SetStateAction<number>>;
  setSoc: Dispatch<SetStateAction<number>>;
  defaultExpanded?: boolean;
}

export const BatteryConfig = (props: BatteryConfigProps) => {
  const { 
    setMaxCRate,
    setPackEnergyCap,
    setPackMaxAh,
    setPackMaxVoltage,
    setPackVoltage,
    setSoh,
    setSoc,
    defaultExpanded,
  } = props

  return (
    <Accordion defaultExpanded={defaultExpanded}>

      <AccordionSummary expandIcon={<ExpandMoreIcon />} style={{ background: '#EEE' }}>
        <Typography>Battery</Typography>
      </AccordionSummary>

      <AccordionDetails>
        <Box width={320} m={'auto'}>

          <TextField
            select
            fullWidth
            size='small'
            variant='filled'
            label='Maximum C-rate'
            helperText='C'
            defaultValue={1}
            onChange={(event) => { setMaxCRate(parseFloat(event.target.value)) }}
          >
            <MenuItem value={0.1}>0.1</MenuItem>
            <MenuItem value={0.25}>0.25</MenuItem>
            <MenuItem value={0.5}>0.5</MenuItem>
            <MenuItem value={0.75}>0.75</MenuItem>
            <MenuItem value={1}>1</MenuItem>
          </TextField>

          <TextField 
            select 
            fullWidth
            size='small' 
            variant='filled'
            label='Energy Capacity'
            helperText='Wh'
            defaultValue={8e4}
            onChange={(event) => { setPackEnergyCap(parseFloat(event.target.value)) }}
          >
            <MenuItem value={8e4}>80,000</MenuItem>
          </TextField>

          <TextField 
            select 
            fullWidth
            size='small' 
            variant='filled' 
            label='Maximum Amp Hours'
            helperText='Ah'
            defaultValue={250}
            onChange={(event) => { setPackMaxAh(parseFloat(event.target.value)) }}
          >
            <MenuItem value={250}>250</MenuItem>
          </TextField>

          <TextField 
            select 
            fullWidth
            size='small' 
            variant='filled' 
            label='Maximum Voltage'
            helperText='V'
            defaultValue={400}
            onChange={(event) => { setPackMaxVoltage(parseFloat(event.target.value)) }}
          >
            <MenuItem value={400}>400</MenuItem>
          </TextField>

          <TextField 
            select 
            fullWidth
            size='small' 
            variant='filled' 
            label='Voltage'
            helperText='V'
            defaultValue={350}
            onChange={(event) => { setPackVoltage(parseFloat(event.target.value)) }}
          >
            <MenuItem value={350}>350</MenuItem>
          </TextField>

          <TextField 
            select 
            fullWidth
            size='small' 
            variant='filled' 
            label='State of Health'
            helperText=' '
            defaultValue={1}
            onChange={(event) => { setSoh(parseFloat(event.target.value)) }}
          >
            <MenuItem value={1}>1</MenuItem>
          </TextField>

          <TextField 
            select 
            fullWidth
            size='small' 
            variant='filled' 
            label='State of Charge'
            defaultValue={0.6}
            onChange={(event) => { setSoc(parseFloat(event.target.value)) }}
          >
            <MenuItem value={0.5}>0.5</MenuItem>
            <MenuItem value={0.6}>0.6</MenuItem>
            <MenuItem value={0.7}>0.7</MenuItem>
            <MenuItem value={0.8}>0.8</MenuItem>
            <MenuItem value={0.9}>0.9</MenuItem>
            <MenuItem value={1}>1</MenuItem>
          </TextField>

        </Box>
      </AccordionDetails>

    </Accordion>
  )
}