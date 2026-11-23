/** @typedef {{key:string,label:string,type:string,required?:boolean,min?:number,max?:number,help?:string,example?:string|number,default?:string|number,choices?:string[]}} Field */
const toValue = (field, value) => {
  if (field.type === 'number' || field.type === 'integer') return Number(value);
  // datetime-local yields a wall time; the desk works in Beijing time (UTC+8).
  if (field.type === 'datetime') return `${value.length === 16 ? `${value}:00` : value}+08:00`;
  return value;
};
export const formValues = (form, fields) => {
  const data = new FormData(form);
  return Object.fromEntries(fields.flatMap(field => {
    const value = data.get(field.key);
    return value === null || value === '' ? [] : [[field.key, toValue(field, value)]];
  }));
};
