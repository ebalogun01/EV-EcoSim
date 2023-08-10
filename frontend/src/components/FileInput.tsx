import React from "react";
import { FormControl, FormHelperText, Input, InputLabel } from "@mui/material";

interface FileInputProps {
  label: string;
  onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  helperText?: string;
  style?: React.CSSProperties;
}

export const FileInput = (props: FileInputProps) => {

  const { label, onChange, helperText, style } = props

  return (
      <FormControl>
        <InputLabel style={{ textAlign: 'left', position: 'relative', width: '100%'}}>{label}</InputLabel>
        <Input size='small' type='file' onChange={onChange} style={style} />
        <FormHelperText>{helperText}</FormHelperText>
      </FormControl>
  )
}