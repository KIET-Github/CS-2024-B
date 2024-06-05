import streamlit as st
import os
import numpy as np
import time
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Embedding, LSTM, Dot, Reshape, Concatenate, BatchNormalization, GlobalMaxPooling2D, Dropout, Add, MaxPooling2D, GRU, AveragePooling2D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
from nltk.translate.bleu_score import sentence_bleu
chexnet_weights = "/content/drive/MyDrive/brucechou1983_CheXNet_Keras_0.3.0_weights (1).h5"

def create_chexnet(chexnet_weights = chexnet_weights, input_size=(224,224)):
    model = tf.keras.applications.DenseNet121(include_top=False, input_shape = input_size+(3,))
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(14, activation="sigmoid", name="chexnet_output")(x)
    chexnet = tf.keras.Model(inputs=model.input, outputs=x)
    chexnet.load_weights(chexnet_weights)
    chexnet = tf.keras.Model(inputs=model.input, outputs=chexnet.layers[-3].output)
    return chexnet

class Image_encoder(tf.keras.layers.Layer):
    def __init__(self, name="image_encoder_block"):
        super().__init__()
        self.chexnet = create_chexnet(input_size=(224,224))
        self.chexnet.trainable = False
        self.avgpool = AveragePooling2D()

    def call(self, data):
        op = self.chexnet(data)
        op = self.avgpool(op)
        op = tf.reshape(op, shape=(-1, op.shape[1]*op.shape[2], op.shape[3]))
        return op

def encoder(image1, image2, dense_dim, dropout_rate):
    im_encoder = Image_encoder()
    bkfeat1 = im_encoder(image1)
    bk_dense = Dense(dense_dim, name='bkdense', activation='relu')
    bkfeat1 = bk_dense(bkfeat1)
    bkfeat2 = im_encoder(image2)
    bkfeat2 = bk_dense(bkfeat2)
    concat = Concatenate(axis=1)([bkfeat1, bkfeat2])
    bn = BatchNormalization(name="encoder_batch_norm")(concat) 
    dropout = Dropout(dropout_rate, name="encoder_dropout")(bn)
    return dropout

class global_attention(tf.keras.layers.Layer):
    def __init__(self, dense_dim):
        super().__init__()
        self.W1 = Dense(units=dense_dim)
        self.W2 = Dense(units=dense_dim)
        self.V = Dense(units=1)

    def call(self, encoder_output, decoder_h):
        decoder_h = tf.expand_dims(decoder_h, axis=1)
        tanh_input = self.W1(encoder_output) + self.W2(decoder_h)
        tanh_output = tf.nn.tanh(tanh_input)
        attention_weights = tf.nn.softmax(self.V(tanh_output), axis=1)
        op = attention_weights * encoder_output
        context_vector = tf.reduce_sum(op, axis=1)
        return context_vector, attention_weights

class One_Step_Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, max_pad, dense_dim, name="onestepdecoder"):
        super().__init__()
        self.embedding = Embedding(input_dim=vocab_size+1, output_dim=embedding_dim, input_length=max_pad, mask_zero=True, name='onestepdecoder_embedding')
        self.LSTM = GRU(units=dense_dim, return_state=True, name='onestepdecoder_LSTM')
        self.attention = global_attention(dense_dim=dense_dim)
        self.final = Dense(vocab_size+1, activation='softmax')

    @tf.function
    def call(self, input_to_decoder, encoder_output, decoder_h):
        embedding_op = self.embedding(input_to_decoder)
        context_vector, attention_weights = self.attention(encoder_output, decoder_h)
        context_vector_time_axis = tf.expand_dims(context_vector, axis=1)
        concat_input = Concatenate(axis=-1)([context_vector_time_axis, embedding_op])
        output, decoder_h = self.LSTM(concat_input, initial_state=decoder_h)
        output = self.final(output)
        return output, decoder_h, attention_weights

class decoder(tf.keras.Model):
    def __init__(self, max_pad, embedding_dim, dense_dim, batch_size, vocab_size):
        super().__init__()
        self.onestepdecoder = One_Step_Decoder(vocab_size=vocab_size, embedding_dim=embedding_dim, max_pad=max_pad, dense_dim=dense_dim)
        self.output_array = tf.TensorArray(tf.float32, size=max_pad)
        self.max_pad = max_pad
        self.batch_size = batch_size
        self.dense_dim = dense_dim
    
    @tf.function
    def call(self, encoder_output, caption):
        decoder_h, decoder_c = tf.zeros_like(encoder_output[:,0]), tf.zeros_like(encoder_output[:,0])
        output_array = tf.TensorArray(tf.float32, size=self.max_pad)
        for timestep in range(self.max_pad):
            output, decoder_h, attention_weights = self.onestepdecoder(caption[:,timestep:timestep+1], encoder_output, decoder_h)
            output_array = output_array.write(timestep, output)

        self.output_array = tf.transpose(output_array.stack(), [1, 0, 2])
        return self.output_array

def create_model():
    input_size = (224, 224)
    tokenizer = joblib.load('/content/drive/MyDrive/Colab Notebooks/tokenizer (4).pkl')
    max_pad = 29
    batch_size = 100
    vocab_size = len(tokenizer.word_index)
    embedding_dim = 300
    dense_dim = 512
    lstm_units = dense_dim
    dropout_rate = 0.2

    image1 = Input(shape=(input_size + (3,)))
    image2 = Input(shape=(input_size + (3,)))
    caption = Input(shape=(max_pad,))

    encoder_output = encoder(image1, image2, dense_dim, dropout_rate)
    output = decoder(max_pad, embedding_dim, dense_dim, batch_size, vocab_size)(encoder_output, caption)
    model = tf.keras.Model(inputs=[image1, image2, caption], outputs=output)
    model_filename = '/content/drive/MyDrive/Encoder_Decoder_global_attention.h5'
    model_save = model_filename
    model.load_weights(model_save)

    return model, tokenizer

def greedy_search_predict(image1, image2, model, tokenizer, input_size=(224,224)):
    image1 = tf.expand_dims(cv2.resize(image1, input_size, interpolation=cv2.INTER_NEAREST), axis=0)
    image2 = tf.expand_dims(cv2.resize(image2, input_size, interpolation=cv2.INTER_NEAREST), axis=0)
    image1 = model.get_layer('image_encoder')(image1)
    image2 = model.get_layer('image_encoder')(image2)
    image1 = model.get_layer('bkdense')(image1)
    image2 = model.get_layer('bkdense')(image2)

    concat = model.get_layer('concatenate')([image1, image2])
    enc_op = model.get_layer('encoder_batch_norm')(concat)  
    enc_op = model.get_layer('encoder_dropout')(enc_op)

    decoder_h, decoder_c = tf.zeros_like(enc_op[:,0]), tf.zeros_like(enc_op[:,0])
    a = []
    pred = []
    max_pad = 29
    for i in range(max_pad):
        if i==0:
            caption = np.array(tokenizer.texts_to_sequences(['<cls>']))
        output, decoder_h, attention_weights = model.get_layer('decoder').onestepdecoder(caption, enc_op, decoder_h)

        max_prob = tf.argmax(output, axis=-1)
        caption = np.array([max_prob])
        if max_prob == np.squeeze(tokenizer.texts_to_sequences(['<end>'])): 
            break
        else:
            a.append(tf.squeeze(max_prob).numpy())
    return tokenizer.sequences_to_texts([a])[0]

def get_bleu(reference, prediction):
    reference = [reference.split()]
    prediction = prediction.split()
    bleu1 = sentence_bleu(reference, prediction, weights=(1,0,0,0))
    bleu2 = sentence_bleu(reference, prediction, weights=(0.5,0.5,0,0))
    bleu3 = sentence_bleu(reference, prediction, weights=(0.33,0.33,0.33,0))
    bleu4 = sentence_bleu(reference, prediction, weights=(0.25,0.25,0.25,0.25))
    return bleu1, bleu2, bleu3, bleu4

def predict1(image1, image2=None, model_tokenizer=None):
    if image2 is None:
        image2 = image1
    if model_tokenizer is None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(image1, image2, model, tokenizer)
    return predicted_caption

def predict2(true_caption, image1, image2=None, model_tokenizer=None):
    if image2 is None:
        image2 = image1
    if model_tokenizer is None:
        model, tokenizer = create_model()
    else:
        model, tokenizer = model_tokenizer[0], model_tokenizer[1]
    predicted_caption = greedy_search_predict(image1, image2, model, tokenizer)
    _ = get_bleu(true_caption, predicted_caption)
    _ = list(_)
    return pd.DataFrame([_], columns=['bleu1', 'bleu2', 'bleu3', 'bleu4'])

def function1(image1, image2, model_tokenizer=None):
    if model_tokenizer is None:
        model_tokenizer = list(create_model())
    predicted_caption = []
    for i1, i2 in zip(image1, image2):
        caption = predict1(i1, i2, model_tokenizer)
        predicted_caption.append(caption)
    return predicted_caption

def function2(true_caption, image1, image2):
    model_tokenizer = list(create_model())
    predicted = pd.DataFrame(columns=['bleu1', 'bleu2', 'bleu3', 'bleu4'])
    for c, i1, i2 in zip(true_caption, image1, image2):
        caption = predict2(c, i1, i2, model_tokenizer)
        predicted = predicted.append(caption, ignore_index=True)
    return predicted

st.set_page_config(page_title='Chest X-ray Report Generator', page_icon=':stethoscope:',initial_sidebar_state="expanded", layout="wide")
st.markdown(
    """
    <style>
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    
    footer {
        background-color: #f0e6ff; /* very light purple */
    }
    .sidebar {
        background-color: #f0e6ff; /* very light purple */
    }
    .sidebar .sidebar-content .block-container {
        background-color: #f0e6ff; /* very light purple */
    }
    .sidebar .sidebar-content .block-container hr {
        background-color: #2e7bcf; /* same as sidebar background */
    }
    .stTextInput>div>div>input {
        background-color: #ffffff; /* white */
        color: #8A2BE2; 
    }
    .stTextInput>div>div>input::placeholder {
        color: #8A2BE2; 
    }
    .stButton>button {
        background-color: #8A2BE2; 
        color: white;
    }
   
    .stSlider>div>div>div>div {
        background-color: #f0e6ff; 
    }
 
    </style>
    """,
    unsafe_allow_html=True
)
custom_css = """
<style>
body {
    background-color:  #f0e6ff; 
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Contents")
page = st.sidebar.radio("Navigate", ["Welcome", "About"])
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: black;
    }
    .sidebar .sidebar-content .sidebar-item {
        color: white;
        font-size: 20px;
    }
    .sidebar .sidebar-content .sidebar-item:hover {
        color: blue;
    }
    .sidebar .sidebar-content .sidebar-item.selected {
        background-color: #02ab21;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Main content
if page == "Welcome":
    st.title("Welcome To The App")
    st.markdown(":point_down:")
    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf, #2e7bcf);
            color: white;
        }
        .sidebar .sidebar-content .block-container {
            background-color: #f0e6ff; /* very light purple */
        }
        .sidebar .sidebar-content .block-container hr {
            background-color: #2e7bcf; /* same as sidebar background */
        }
        .sidebar .sidebar-content .element-container {
            padding: 10px;
        }
        .sidebar .sidebar-content .element-container .fullScreenFrame{
            color: #8A2BE2;
            background-color: #ffffff; /* white */
        }
        .sidebar .sidebar-content .element-container .fullScreenFrame:hover{
            background-color: #8A2BE2;
            color: white;
        }
        .sidebar .sidebar-content .element-container .stButton>button {
            background-color: #8A2BE2;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
       f"""
        <div style="color: #A855F7; font-size: 50px; font-weight: bold; display: flex; justify-content: center; align-items: center; height: 5vh;">
            Chest X-ray Report Generator
        </div>
        <div style="display: flex; justify-content: center; align-items: center; font-size: 15px; font-weight: bold; color: black;height: 10vh;">
            Provides you customized reports
        </div>
        """,
        unsafe_allow_html=True
    )
    bg_image = Image.open("/content/drive/MyDrive/Colab Notebooks/web_app_img1.jpg")
    st.image(bg_image, use_column_width=True)
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("/content/drive/MyDrive/Colab Notebooks/web_app_img1.jpg") no-repeat center center fixed;
            background-size: cover;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("\nThis app will generate the impression part of an X-ray report.\nYou can upload 2 X-rays that are front view and side view of the chest of the same individual.")
    st.markdown("The 2nd X-ray is optional.")
    col1, col2 = st.columns(2)
    image_1 = col1.file_uploader("X-ray 1", type=['png', 'jpg', 'jpeg'])
    image_2 = None
    if image_1:
        image_2 = col2.file_uploader("X-ray 2 (optional)", type=['png', 'jpg', 'jpeg'])

    col1, col2 = st.columns(2)
    predict_button = col1.button('Predict on uploaded files')
    test_data = col2.button('Predict on sample data')

    def predict(image_1, image_2, model_tokenizer, predict_button=predict_button):
        if predict_button:
            if image_1 is not None:
                start = time.process_time()
                image_1 = Image.open(image_1).convert("RGB")
                image_1 = np.array(image_1) / 255.0
                if image_2 is None:
                    image_2 = image_1
                else:
                    image_2 = Image.open(image_2).convert("RGB")
                    image_2 = np.array(image_2) / 255.0
                st.image([image_1, image_2], width=300)
                predicted_caption = predict1(image_1, image_2, model_tokenizer)
                st.markdown("### **Impression:**")
                impression = st.empty()
                impression.write(predicted_caption)
                time_taken = f"Time Taken for prediction: {time.process_time() - start:.2f} seconds"
                st.write(time_taken)
            else:
                st.markdown("## Upload an Image")

    def predict_sample(model_tokenizer, folder='/content/drive/MyDrive/MedGen-main/data/sample'):
        no_files = len(os.listdir(folder))
        file = np.random.randint(1, no_files)
        file_path = os.path.join(folder, str(file))
        if len(os.listdir(file_path)) == 2:
            image_1 = os.path.join(file_path, os.listdir(file_path)[0])
            image_2 = os.path.join(file_path, os.listdir(file_path)[1])
        else:
            image_1 = os.path.join(file_path, os.listdir(file_path)[0])
            image_2 = image_1
        predict(image_1, image_2, model_tokenizer, True)

    # Load the model and tokenizer
    model_tokenizer = create_model()

    # Conditional execution based on button clicks
    if test_data:
        predict_sample(model_tokenizer)
    elif predict_button:
        predict(image_1, image_2, model_tokenizer) 










elif page == "About":
    st.title("About")
st.markdown("\n### About the App")
st.markdown("This app is designed to generate the impression part of an X-ray report.")
st.markdown("It uses a neural network model to analyze uploaded X-ray images and produce the report.")

st.markdown("\n### Frequently Asked Questions (FAQ)")

st.markdown("- **How to click a Photo/Image of your X-Ray using your smartphone camera for better results?**")
st.markdown("     **Answer:** Place your x-ray in front of a white screen (e.g., TVs, Laptop screen, etc., where dim white light is coming from it) so that the content of the X-Ray is clearly visible. Then, click a clear image of your x-ray. (Warning: Don't place your X-Ray film directly in bright light like sunlight. It may damage the X-Ray.)")

st.markdown("-**How to upload the chest x-ray image?**")
st.markdown("   **Answer:** Users can directly upload the image from the system. Users can upload two images at a time.")

st.markdown("- **Do the app accept images in all formats?**")
st.markdown("   **Answer:** Yes, our app accepts images in different formats (.jpg, .jpeg, .png) to give convenience to the users.")

st.markdown("\n### Get In Touch")
st.markdown("For any inquiries or feedback, please contact us at:")
st.markdown("  Phone: +91-12345678")
st.markdown("  Alternate Email 1: anmolratan2411@gmail.com")
st.markdown("  Alternate Email 2: 1997anssingh@gmail.com")
