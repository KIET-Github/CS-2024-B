//organizerAddress = '0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266'

import voting from "./Create.json";

export const VotingAddress = '0x5C952567439A46a160050A19A77aBd67EDDad084';
export const VotingAddressABI = voting.abi;
export const organiserAddress = (process.env.NEXT_PUBLIC_ORGANISER).toLowerCase();