//INTERNAL IMPORT
import Style from './RemoveButton.module.css'

const RemoveButton = ({ classStyles, btnName, handleClick }) => (
    <button className={Style.button} type="button" onClick={handleClick}>
        {btnName}
    </button>
);

export default RemoveButton;
