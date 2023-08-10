
import { Dispatch, SetStateAction } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Box, MenuItem, TextField, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { FileInput } from './FileInput';

interface SolarConfigProps {
  setFile: Dispatch<SetStateAction<File | undefined>>;
  setStartYear: Dispatch<SetStateAction<number>>;
  setStartMonth: Dispatch<SetStateAction<number>>
  defaultExpanded?: boolean;
}

export const SolarConfig = (props: SolarConfigProps) => {
  const { 
    setFile,
    setStartYear,
    setStartMonth,
    defaultExpanded } = props

  return (
    <Accordion defaultExpanded={defaultExpanded}>

      <AccordionSummary expandIcon={<ExpandMoreIcon />} style={{ background: '#EEE' }}>
        <Typography>Solar</Typography>
      </AccordionSummary>

      <AccordionDetails>
        <Box width={320} m='auto'>

          <FileInput 
            label='Upload solar data:' 
            onChange={ (event) => {
              const file = event.target.files && event.target.files[0] 
              if (file) {
                setFile(file)
              }
            }}
            helperText='Default is San Jose temperature data starting from January 2018'
            style={{ paddingBottom: '12px' }}
          />

          <TextField
            fullWidth
            size='small'
            variant='filled' 
            label='Start Year'
            helperText=' '
            defaultValue='2018'
            onChange={(event) => { setStartYear(parseFloat(event.target.value)) }}
          >
          </TextField>

          <TextField
            select
            fullWidth
            size='small'
            variant='filled' 
            label='Start Month'
            helperText=' '
            defaultValue={1}
            onChange={(event) => { setStartMonth(parseFloat(event.target.value)) }}
          >
            <MenuItem value={1}>January</MenuItem>
            <MenuItem value={2}>February</MenuItem>
            <MenuItem value={3}>March</MenuItem>
            <MenuItem value={4}>April</MenuItem>
            <MenuItem value={5}>May</MenuItem>
            <MenuItem value={6}>June</MenuItem>
            <MenuItem value={7}>July</MenuItem>
            <MenuItem value={8}>August</MenuItem>
            <MenuItem value={9}>September</MenuItem>
            <MenuItem value={10}>October</MenuItem>
            <MenuItem value={11}>November</MenuItem>
            <MenuItem value={12}>December</MenuItem>
          </TextField>

        </Box>
      </AccordionDetails>

    </Accordion>
  )
}