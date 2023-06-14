from pathlib import Path
import torch
# from st_on_hover_tabs import on_hover_tabs
import streamlit as st
st.set_page_config(layout="wide")

model_path = './model.iter-685000'

import sys, os
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import Descriptors
import sascorer
import networkx as nx
from stqdm import stqdm
import base64, io
import pandas as pd
import streamlit_ext as ste

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sys.path.append('%s/fast_jtnn/' % os.path.dirname(os.path.realpath(__file__)))
from mol_tree import Vocab, MolTree
from jtprop_vae import JTPropVAE
from molbloom import buy

css='''
[data-testid="metric-container"] {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
}
[data-testid="stDataFrameResizable"] {
    width: fit-content;
    margin: auto;
}
[data-testid="stSidebar"]{
        max-width: 300px;
    }
'''

st.markdown(f'<style>{css}</style>',unsafe_allow_html=True)

s_buff = io.StringIO()
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path,max_width=500):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='max-width: {}px;'>".format(
      img_to_bytes(img_path), max_width
    )
    return img_html

_mcf = pd.read_csv('./mcf.csv')
_pains = pd.read_csv('./wehi_pains.csv',
                     names=['smarts', 'names'])
_mcf_filters = [Chem.MolFromSmarts(x) for x in
            _mcf['smarts'].values]
_pains_filters = [Chem.MolFromSmarts(x) for x in
            _pains['smarts'].values]

def mol_passes_filters_custom(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    if mol is None:
        return 'NoMol'
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return 'ManyRings'
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return 'Charged'
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return 'AtomNotAllowed'
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _mcf_filters):
        return 'MCF'
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _pains_filters):
        return 'PAINS'
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return 'Isomeric'
    if Chem.MolFromSmiles(smiles) is None:
        return 'Isomeric'
    if not check_vocab(Chem.MolToSmiles(mol)):
        return 'NoVocab'
    return 'YES'

def penalized_logp_standard(mol):

    logP_mean = 2.4399606244103639873799239
    logP_std = 0.9293197802518905481505840
    SA_mean = -2.4485512208785431553792478
    SA_std = 0.4603110476923852334429910
    cycle_mean = -0.0307270378623088931402396
    cycle_std = 0.2163675785228087178335699

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length
    # print(logP_mean)

    standardized_log_p = (log_p - logP_mean) / logP_std
    standardized_SA = (SA - SA_mean) / SA_std
    standardized_cycle = (cycle_score - cycle_mean) / cycle_std
    return log_p,SA,cycle_score,standardized_log_p + standardized_SA + standardized_cycle

def df_to_file(df):
    s_buff.seek(0)
    df.to_csv(s_buff)
    return s_buff.getvalue().encode()

def download_df(df,id):
    with st.expander(':arrow_down: Download this dataframe'):
        st.markdown("<h4 style='color:tomato;'>Select column(s) to save:</h4>",unsafe_allow_html=True)
        for col in df.columns:
            st.checkbox(col,key=str(id)+'_col_'+str(col))
        st.text_input('File name (.csv):','dataframe',key=str(id)+'_file_name')
        
        ste.download_button('Download',df_to_file(df[[col for col in df.columns if st.session_state[str(id)+'_col_'+str(col)]]]),st.session_state[str(id)+'_file_name']+'.csv')

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

    
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 0
    
if 'current_step' not in st.session_state:
    st.session_state['current_step'] = 0
    
def set_page_view(id):
    st.session_state['current_view'] = id
    
def get_page_view():
    return st.session_state['current_view']
    
def set_step(id):
    st.session_state['current_step'] = id
    
def get_step():
    return st.session_state['current_step']


vocab = [x.strip("\r\n ") for x in open('./vocab.txt')] 
vocab_set = set(vocab)
vocab = Vocab(vocab)

def check_vocab(smiles):
    cset = set()
    mol = MolTree(smiles)
    for c in mol.nodes:
        cset.add(c.smiles)
    return cset.issubset(vocab_set)


@st.cache_resource
def load_model():
    model = JTPropVAE(vocab, 450, 56, 20, 3)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model.to('cuda')
    else:
        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    return model

from streamlit_lottie import st_lottie
import requests
    
def render_animation():
    animation_response = requests.get('https://assets1.lottiefiles.com/packages/lf20_vykpwt8b.json')
    animation_json = dict()
    
    if animation_response.status_code == 200:
        animation_json = animation_response.json()
    else:
        print("Error in the URL")
    return st_lottie(animation_json,height=200,width=300)

def oam_sidebar(step):
    st.title('**Optimize a molecule**')
    prog_bar = st.progress(0)
    # cur_step = get_step()
    if step == 0: prog_bar.progress(0)
    if step == 1: prog_bar.progress(33)
    if step == 2: prog_bar.progress(67)
    if step == 3: prog_bar.progress(100)
    st.markdown('\n')
    
    # st.markdown(get_step())
    color_ls = colorize_step(4,step)
    
    st.markdown("<h4 style='color: "+color_ls[0]+"'>Choose a molecule</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[1]+"'>Choose settings</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[2]+"'>Optimizing a molecule</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[3]+"'>Finished</h4>",unsafe_allow_html=True)

def oab_sidebar(step):
    st.title('**Optimize a batch**')
    prog_bar = st.progress(0)
    # cur_step = get_step()
    if step == 0: prog_bar.progress(0)
    if step == 1: prog_bar.progress(20)
    if step == 2: prog_bar.progress(40)
    if step == 3: prog_bar.progress(60)
    if step == 4: prog_bar.progress(80)
    if step == 5: prog_bar.progress(100)
    st.markdown('\n')
    
    # st.markdown(get_step())
    color_ls = colorize_step(6,step)
    
    st.markdown("<h4 style='color: "+color_ls[0]+"'>Upload SMILES file</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[1]+"'>Checking SMILES</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[2]+"'>Select scores</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[3]+"'>Choose settings</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[4]+"'>Optimizing a batch</h4>",unsafe_allow_html=True)
    st.markdown('|')
    st.markdown("<h4 style='color: "+color_ls[5]+"'>Finished</h4>",unsafe_allow_html=True)

# @st.cache_data(experimental_allow_widgets=True)

# if 'sidebar_con' not in st.session_state:
#     sidebar_con = st.empty()
# def render_sidebar(page,step):
#     sidebar_con.empty()
#     with sidebar_con.container():
#         if page == 0:
#             with st.sidebar():
#                 oam_sidebar(step)

def colorize_step(n_step,cur_step):
    color_list = ['grey']*n_step
    for i in range(cur_step):
        color_list[i] = 'mediumseagreen'
    color_list[cur_step] = 'tomato'
    if cur_step == n_step-1:
        color_list[cur_step] = 'mediumseagreen'
    return color_list

def form_header():
    st.markdown("<h1 style='padding: 25px;text-align: center;color: white;background-color: tomato;'>Molecular Optimization using Junction Tree Variational Autoencoder</h1>",unsafe_allow_html=True)
    st.markdown("<h4 style='padding: 10px;text-align: center;color: white;background-color: mediumseagreen;'>Gia-Bao Truong</h4>",unsafe_allow_html=True)
    with st.expander(':star2: About the model'):
        st.markdown("<p style='text-align: center;'>Based on Junction Tree Variational Autoencoder for Molecular Graph Generation (JTVAE)</p>",unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Wengong Jin, Regina Barzilay, Tommi Jaakkola</p>",unsafe_allow_html=True)

    # determines button color which should be red when user is on that given step
    oam_type = 'primary' if st.session_state['current_view'] == 0 else 'secondary'
    oab_type = 'primary' if st.session_state['current_view'] == 1 else 'secondary'
    ab_type = 'primary' if st.session_state['current_view'] == 2 else 'secondary'

    step_cols = st.columns([.2,.85,.85,.85,.2])    
    step_cols[1].button('Optimize a molecule',on_click=set_page_view,args=[0],type=oam_type,use_container_width=True)
    step_cols[2].button('Optimize a batch',on_click=set_page_view,args=[1],type=oab_type,use_container_width=True)        
    step_cols[3].button('About',on_click=set_page_view,args=[2],type=ab_type,use_container_width=True) 
    st.empty()
    
def form_body():
    body_container = st.empty()
    ###### Optimize a molecule ######
    if st.session_state['current_view'] == 0:
        body_container.empty()
        with body_container.container():
            Optimize_a_molecule()
    ###### Optimize a batch ######
    if st.session_state['current_view'] == 1: 
        body_container.empty()
        with body_container.container():
            Optimize_a_batch()
    ###### About ######
    if st.session_state['current_view'] == 2:
        body_container.empty()
        with body_container.container():
            About()
        
def About():
    descrip_model = '''
We seek to automate the design of molecules based on specific chemical properties. In computational terms, this task involves continuous embedding and generation of molecular graphs. Our primary contribution is the direct realization of molecular graphs, a task previously approached by generating linear SMILES strings instead of graphs. Our junction tree variational autoencoder generates molecular graphs in two phases, by first generating a tree-structured scaffold over chemical substructures, and then combining them into a molecule with a graph message passing network. This approach allows us to incrementally expand molecules while maintaining chemical validity at every step. We evaluate our model on multiple tasks ranging from molecular generation to optimization. Across these tasks, our model outperforms previous state-of-the-art baselines by a significant margin.
'''
    img_caption = '''
Figure 3. Overview of our method: A molecular graph G is first decomposed into its junction tree TG, where each colored node in the tree represents a substructure in the molecule. We then encode both the tree and graph into their latent embeddings zT and zG. To decode the molecule, we first reconstruct junction tree from zT , and then assemble nodes in the tree back to the original molecule.'''
    
    with st.expander(':four_leaf_clover: About the author',expanded=True):
        st.markdown("<h4 style='text-align:center;'>Gia-Bao Truong</h4>",unsafe_allow_html=True)
        st.markdown("<h4 style='color:tomato; text-align:center;'>Student at</h4>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>"+
                        img_to_html('img/about1.png',64)+' '+img_to_html('img/about2.png',64)+
                        "</p>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>Faculty of Pharmacy, University of Medicine and Pharmacy at Ho Chi Minh City</h5>",unsafe_allow_html=True)
        st.markdown("<h4 style='color:tomato; text-align:center;'>Team</h4>",unsafe_allow_html=True)
        st.markdown("<p style='text-align:center;'>"+
                        img_to_html('img/about3.png',64)+
                        "</p>", unsafe_allow_html=True)
        st.markdown("<h5 style='text-align:center;'>MedAI</h5>",unsafe_allow_html=True)
        
    
    with st.expander(':star2: About the model',expanded=True):
        st.markdown("Based on Junction Tree Variational Autoencoder for Molecular Graph Generation (JTVAE)",unsafe_allow_html=True)
        st.markdown("<h4 style='color:tomato;'>Citing</h4>",unsafe_allow_html=True)
        st.markdown("Paper: [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)")
        st.code('''@misc{jin2019junction,
                        title={Junction Tree Variational Autoencoder for Molecular Graph Generation}, 
                        author={Wengong Jin and Regina Barzilay and Tommi Jaakkola},
                        year={2019},
                        eprint={1802.04364},
                        archivePrefix={arXiv},
                        primaryClass={cs.LG}
                        }''')
        st.markdown("<h4 style='color:tomato;'>Author</h4>",unsafe_allow_html=True)
        st.markdown("Wengong Jin, Regina Barzilay, Tommi Jaakkola",unsafe_allow_html=True)
        st.markdown("<h4 style='color:tomato;'>Abstract</h4>",unsafe_allow_html=True)
        st.markdown(descrip_model)
        ab = st.columns([1,10,1])
        ab[1].markdown("<p style='text-align: center;'>"+
                        img_to_html('img/model_fig.png')+
                        "</p>", unsafe_allow_html=True)
        ab[1].markdown("<p style='text-align: center;'>"+
                        img_caption+
                        "</p>",unsafe_allow_html=True)
    
def Optimize_a_molecule():
    st.markdown("<h2 style='text-align: center;'>Optimize a molecule</h2>",unsafe_allow_html=True)
    with st.expander(':snowman: :blue[Instruction]'):
        guide = """<h4 style='color:tomato;'>Steps to optimize a molucule</h4>
        1. Select from examples, or manually enter a valid SMILES string of a molecule.</br>
        2. Configure the settings to generate a new molecule. The new molecule should have a higher penalized LogP value.</br>
            - Learning rate: How 'far' from the molecule that you want to search.</br>
            - Similarity cutoff: How 'similar' to the molecule that you want to search.</br>
            - Number of iterations: Number of generation trials.</br>
        <h4 style='color:darkturquoise;'>Annotation</h4>
        <b>SMILES</b> - Simplified molecular-input line-entry system</br>
        <b>LogP</b> - The log of the partition coefficient of a solute between octanol and water, at near infinite dilution</br>
        <b>SA score</b> - Synthetic Accessibility Score (lower is better)</br>
        <b>Cycle score</b> - A number of carbon rings of size larger than 6 (lower is better)</br>
        <b>Penalized LogP</b> - Standardized score of <i>LogP - SA score - Cycle score</i></br>
        <b>Similarity</b> - Molecular similarity is calculated via Morgan fingerprint of radius 2 with Tanimoto similarity</br>
        """
        st.markdown(guide,unsafe_allow_html=True)
        
    with st.sidebar:
        sidebar_con = st.empty()
    # sidebar_con.empty()
    with sidebar_con.container():
            set_step(0)
            oam_sidebar(0)
    # oab_sel_container = st.empty()
    if 'checked_single' not in st.session_state:
        st.session_state.checked_single = 'NO'
    # if 'mode' not in st.session_state:
    #     st.session_state.mode = 0
    if 'single_optimized' not in st.session_state:
        st.session_state.single_optimized = False
    if 'smiles_checked' not in st.session_state:
        st.session_state.smiles_checked = False
    # with oab_sel_container.container():
    ls_opt = ['-','Sorafenib','Pazopanib','Sunitinib']
    sample_mode = {
        '-':'',
        'Sorafenib':'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',
        'Pazopanib':'CC1=C(C=C(C=C1)NC2=NC=CC(=N2)N(C)C3=CC4=NN(C(=C4C=C3)C)C)S(=O)(=O)N',
        'Sunitinib':'CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C'
        }
    oam_sel_col = st.columns([3,7])
    with st.form('sel_smiles'):
        mode = oam_sel_col[0].selectbox("Select an example",options=ls_opt,on_change=reset_oam_state)
        smiles = oam_sel_col[1].text_input('Enter a SMILES string (max 200 chars):',sample_mode[mode],max_chars=200,
                                  disabled=(mode != '-'))
        # if mode == '-':
        #     st.session_state.smiles = oam_sel_col[1].text_input('Enter a SMILES string (max 200 chars):',max_chars=200,key='opt_0')
        #     # st.session_state.mode = 0
        # elif mode == 'Sorafenib':
        #     st.session_state.smiles = 'CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F'
        #     oam_sel_col[1].text_input('Enter a SMILES string (max 200 chars):','CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F',max_chars=200,disabled=True,key='opt_1')
        #     # st.session_state.mode = 1
        # elif mode == 'Pazopanib':
        #     st.session_state.smiles = 'CC1=C(C=C(C=C1)NC2=NC=CC(=N2)N(C)C3=CC4=NN(C(=C4C=C3)C)C)S(=O)(=O)N'
        #     oam_sel_col[1].text_input('Enter a SMILES string (max 200 chars):','CC1=C(C=C(C=C1)NC2=NC=CC(=N2)N(C)C3=CC4=NN(C(=C4C=C3)C)C)S(=O)(=O)N',max_chars=200,disabled=True,key='opt_2')
        #     # st.session_state.mode = 2
        # elif mode == 'Sunitinib':
        #     st.session_state.smiles = 'CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C'
        #     oam_sel_col[1].text_input('Enter a SMILES string (max 200 chars):','CCN(CC)CCNC(=O)C1=C(NC(=C1C)C=C2C3=C(C=CC(=C3)F)NC2=O)C',max_chars=200,disabled=True,key='opt_3')
        #     # st.session_state.mode = 3
        check_single_butt = st.form_submit_button('Check SMILES',use_container_width=True)
    # st.session_state.smiles = st.session_state['opt_'+str(ls_opt.index(st.session_state.mode))]
    if check_single_butt:
        st.session_state.mode = mode
        st.session_state.smiles = smiles
        check_single(st.session_state.smiles)
    
    if 'optim_single_butt' not in locals():
        optim_single_butt = False
    
    check_single_con = st.empty()
    if 'smiles_selected' in st.session_state:
        if st.session_state.smiles_selected:
            with check_single_con.container():
                if 'checked_single' in st.session_state:
                    if st.session_state.checked_single == 'EnterError':
                        st.markdown("<p style='text-align: center; color: red;'><b>Please enter a SMILES string.</b></p>",unsafe_allow_html=True)
                        # sidebar_con.empty()
                        with sidebar_con.container():
                                set_step(0)
                                oam_sidebar(0)
                    elif st.session_state.checked_single == 'MolError':
                        st.markdown("<p style='text-align: center; color: red;'><b>SMILES is invalid. Please enter a valid SMILES.</b></p>",unsafe_allow_html=True)
                        # sidebar_con.empty()
                        with sidebar_con.container():
                                set_step(0)
                                oam_sidebar(0)
                    elif st.session_state.checked_single == 'YES':
                        if st.session_state.mode != '-':
                            st.markdown(f"<h4 style='color:mediumseagreen;'>Using example: <b>{st.session_state.mode}</b></h4>",unsafe_allow_html=True)
                        else:
                            st.markdown(f"<h4>Selected SMILES</h4>",unsafe_allow_html=True)
                            st.code(st.session_state.smiles)
                        st.markdown("<b>Canonicalized SMILES</b>",unsafe_allow_html=True)
                        st.session_state.canon_smiles = Chem.CanonSmiles(st.session_state.smiles)
                        st.code(st.session_state.canon_smiles)
                        st.markdown("<p style='text-align: center; color: mediumseagreen'>MOSES filters passed successfully.</p>",unsafe_allow_html=True)
                        mol = Chem.MolFromSmiles(st.session_state.canon_smiles)
                        imgByteArr = io.BytesIO()
                        MolToImage(mol,size=(400,400)).save(imgByteArr,format='PNG')
                        st.markdown("<p style='text-align: center;'>"+
                                f"<img src='data:image/png;base64,{base64.b64encode(imgByteArr.getvalue()).decode()}' class='img-fluid'>"+
                                "</p>", unsafe_allow_html=True)
                        # st.image(MolToImage(mol,size=(300,300)))
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric('LogP', '%.5f' % (st.session_state.logp))
                        col2.metric('SA score', '%.5f' % (-st.session_state.sa))
                        col3.metric('Cycle score', '%d' % (-st.session_state.cycle))
                        col4.metric('Penalized LogP', '%.5f' % (st.session_state.pen_p))
                        
                        st.session_state.smiles_checked = True
                        # render_sidebar()
                        # col1, col2, col3 = st.columns(3)
                        # sidebar_con.empty()
                        with sidebar_con.container():
                                set_step(1)
                                oam_sidebar(1)
                        with st.form(":gear: Settings"):
                                st.slider('Choose learning rate: ',0.0,5.0,0.4,key='lr_s')
                                st.slider('Choose similarity cutoff: ',0.0,1.0,0.4,key='sim_cutoff_s')
                                st.slider('Choose number of iterations: ',1,100,80,key='n_iter_s')
                                optim_single_butt = st.form_submit_button("Optimize")
                    else:
                        st.markdown("<b>Canonicalized SMILES</b>",unsafe_allow_html=True)
                        st.code(st.session_state.canon_smiles)
                        if st.session_state.checked_single == 'NoVocab':
                            st.markdown("<p style='text-align: center; color: red;'><b>The molecule contains unavailable vocab(s). Please use another molecule.</b></p>",unsafe_allow_html=True)
                        else:
                            st.markdown("<p style='text-align: center; color: red;'><b>MOSES filters passed failed. Please use another molecule.</b></p>",unsafe_allow_html=True)
                        # sidebar_con.empty()
                        with sidebar_con.container():
                                set_step(0)
                                oam_sidebar(0)
        else: check_single_con.empty()
    
    optim_single_con = st.empty()
    if st.session_state.smiles_checked:
        if optim_single_butt:
            # sidebar_con.empty()
            with sidebar_con.container():
                    set_step(2)
                    oam_sidebar(2)
            
            ani_con = st.empty()
            with ani_con.container():
                st.markdown('Operation in progress. Please wait...')
                render_animation()
                model = load_model()
                st.session_state.new_smiles,st.session_state.sim = optim_single(st.session_state.canon_smiles,model,st.session_state.lr_s,st.session_state.sim_cutoff_s,st.session_state.n_iter_s)
                st.session_state.single_optimized = True
            ani_con.empty()
                # sidebar_con.empty()
        if st.session_state.single_optimized:
            with optim_single_con.container():
                if st.session_state.new_smiles is None:
                    st.markdown("<h4 style='text-align: center; color: red;'>Cannot optimize! Please choose another setting.</h4>",unsafe_allow_html=True)
                else:
                    st.markdown("<b style='text-align: center;'>New SMILES</b>",unsafe_allow_html=True)
                    st.code(st.session_state.new_smiles)
                    new_mol = Chem.MolFromSmiles(st.session_state.new_smiles)
                    if new_mol is None:
                        st.markdown("<p style='text-align: center; color: red;'>New SMILES is invalid! Please choose another setting.</p>",unsafe_allow_html=True)
                        # st.write('New SMILES is invalid.')
                    else:
                        # st.write('New SMILES molecule:')
                        imgByteArr = io.BytesIO()
                        MolToImage(new_mol,size=(400,400)).save(imgByteArr,format='PNG')
                        st.markdown("<p style='text-align: center;'>"+
                                    f"<img src='data:image/png;base64,{base64.b64encode(imgByteArr.getvalue()).decode()}' class='img-fluid'>"+
                                    "</p>", unsafe_allow_html=True)
                        
                        new_moses_passed = mol_passes_filters_custom(new_mol)
                        if new_moses_passed=='YES':
                            st.markdown("<p style='text-align: center; color: mediumseagreen'>MOSES filters passed successfully.</p>",unsafe_allow_html=True)
                        else:
                            st.markdown("<p style='text-align: center; color: red;'><b>MOSES filters passed failed.</b></p>",unsafe_allow_html=True)
                        st.session_state.new_logp,st.session_state.new_sa,st.session_state.new_cycle,st.session_state.new_pen_p = penalized_logp_standard(new_mol)
                    # st.write('New penalized logP score: %.5f' % (new_score))
                        col12, col22, col32, col42 = st.columns(4)
                        col12.metric('LogP', '%.5f' % (st.session_state.new_logp),'%.5f'%(st.session_state.new_logp-st.session_state.logp))
                        col22.metric('SA score', '%.5f' % (-st.session_state.new_sa),'%.5f'%(-st.session_state.new_sa+st.session_state.sa),delta_color='inverse')
                        col32.metric('Cycle score', '%d' % (-st.session_state.new_cycle),'%d'%(-st.session_state.new_cycle+st.session_state.cycle),delta_color='inverse')
                        col42.metric('Penalized LogP', '%.5f' % (st.session_state.new_pen_p),'%.5f'%(st.session_state.new_pen_p-st.session_state.pen_p))
                        # st.metric('New penalized logP score','%.5f' % (new_score), '%.5f'%(new_score-score))
                        st.metric('Similarity','%.5f' % (st.session_state.sim))
                        # st.write('Caching ZINC20 if necessary...')
                        with st.spinner("Caching ZINC20 if necessary..."):
                            if buy(st.session_state.new_smiles, catalog='zinc20',canonicalize=True):
                                st.write('This molecule exists.')
                                st.markdown("<h3 style='text-align: center; color: darkturquoise;'><b>This molecule exists.</h3>",unsafe_allow_html=True)
                            else:
                                # st.write('THIS MOLECULE DOES NOT EXIST!')
                                st.markdown("<h3 style='text-align: center; color: mediumseagreen;'>THIS MOLECULE DOES NOT EXIST!</h3>",unsafe_allow_html=True)
                        st.markdown("<p style='text-align: center; color: grey;'>Checked using molbloom</p>",unsafe_allow_html=True)
            with sidebar_con.container():
                    set_step(3)
                    oam_sidebar(3)
        else: optim_single_con.empty()
    else: optim_single_con.empty()


def check_single(smiles):
    # render_view()
    st.session_state.smiles_selected = True
    # st.session_state.smiles = smiles
    # check_single_con = st.empty()
    
    # optim = False
    # with check_single_con.container():
    if len(smiles) == 0:
        st.session_state.checked_single = 'EnterError'
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.session_state.checked_single = 'MolError'
        else:
            st.session_state.canon_smiles = Chem.MolToSmiles(mol)
            st.session_state.logp,st.session_state.sa,st.session_state.cycle,st.session_state.pen_p = penalized_logp_standard(mol)
            moses_passed = mol_passes_filters_custom(mol)
            st.session_state.checked_single = moses_passed


def optim_single(smiles,model,lr,sim_cutoff,n_iter):

        new_smiles,sim = model.optimize(smiles, sim_cutoff=sim_cutoff, lr=lr, num_iter=n_iter)

        return new_smiles,sim

        
            
def Optimize_a_batch():
    st.session_state.sc_name = ['logp','sa','cycle','pen_logp']
    st.session_state.new_sc_name = ['new_'+n for n in st.session_state.sc_name]
    st.markdown("<h2 style='text-align: center;'>Optimize a batch</h2>",unsafe_allow_html=True)
    with st.expander(':snowman: :blue[Instruction]'):
        guide = """<h4 style='color:tomato;'>Steps to optimize a molucule</h4>
        1. Upload a text file with SMILES string on each line.</br>
        2. Check the SMILES strings to make sure that they are valid and pass MOSES filters.</br>
        3. Select scores to calculate (penalized LogP included). Keep passed SMILES and calculate selected scores.</br>
        4. Configure the settings to generate new molecules. The new molecules should have higher penalized LogP values.</br>
            - Learning rate: How 'far' from each molecule that you want to search</br>
            - Similarity cutoff: How 'similar' to each molecule that you want to search</br>
            - Number of iterations: Number of generation trials per molecule</br>
        5. <i>(Optional)</i> You can download the dataframe at any steps as *.csv file.</br>
        <h4 style='color:darkturquoise;'>Annotation</h4>
        <b>SMILES</b> - Simplified molecular-input line-entry system</br>
        <b>LogP</b> - The log of the partition coefficient of a solute between octanol and water, at near infinite dilution</br>
        <b>SA score</b> - Synthetic Accessibility Score (lower is better)</br>
        <b>Cycle score</b> - A number of carbon rings of size larger than 6 (lower is better)</br>
        <b>Penalized LogP</b> - Standardized score of <i>LogP - SA score - Cycle score</i></br>
        <b>Similarity</b> - Molecular similarity is calculated via Morgan fingerprint of radius 2 with Tanimoto similarity</br>
        """
        st.markdown(guide,unsafe_allow_html=True)
        
    with st.sidebar:
        sidebar_con = st.empty()
    # sidebar_con.empty()
    with sidebar_con.container():
            set_step(0)
            oab_sidebar(0)
    oab_upl_container = st.empty()
    if 'smiles_upload_change' not in st.session_state:
        st.session_state.smiles_upload_change = False
    if 'checked_batch' not in st.session_state:
        st.session_state.checked_batch = False
    if 'batch_left_checked' not in st.session_state:
        st.session_state.batch_left_checked = False
    if 'scores_calculated' not in st.session_state:
        st.session_state.scores_calculated = False
    if 'batch_optimized' not in st.session_state:
        st.session_state.batch_optimized = False
    
    with oab_upl_container.container():
        st.session_state['smiles_file'] = st.file_uploader("Upload a text file with SMILES on each line :sparkles:",on_change=reset_oab_state)
    if 'check_batch_butt' not in locals():
        check_batch_butt = False
    
    if st.session_state['smiles_file'] is not None:
        if st.session_state.smiles_upload_change:
            smiles_list = io.StringIO(st.session_state.smiles_file.getvalue().decode("utf-8"))
            smiles_list = list(smiles_list.getvalue().rstrip().split('\n'))
            st.markdown('Number of SMILES: '+str(len(smiles_list)))
            if len(smiles_list) == 1:
                    st.markdown("<p style='text-align: center; color: red;'><b>Please use <i>Optimize a molecule</i> tab.</b></p>",unsafe_allow_html=True)
                    with sidebar_con.container():
                        set_step(0)
                        oab_sidebar(0)
            else:
                st.session_state['df'] = pd.DataFrame({'SMILES':smiles_list})
                st.dataframe(st.session_state['df'],use_container_width=True)
                check_batch_butt = st.button('Check SMILES')
        else:
            # if not st.session_state.checked_batch:
            if st.session_state['smiles_file'] is not None:
                st.dataframe(st.session_state['df'],use_container_width=True)
                # st.button('Check SMILES',on_click=check_batch,args=[smiles_list],key='check_batch_butt')

    if check_batch_butt:
        if st.session_state.smiles_upload_change:
            with sidebar_con.container():
                    set_step(1)
                    oab_sidebar(1)
            check_batch(list(st.session_state['df'].SMILES))
            st.session_state.smiles_upload_change = False

    if 'calc_batch_butt' not in locals():
        calc_batch_butt = False
    check_batch_con = st.empty()
    calc_batch_con = st.empty()
    if st.session_state.checked_batch:
        with check_batch_con.container():
            passed_num = st.session_state['df'][st.session_state['df'].checked != 'invalid'].shape[0]
            st.markdown('Number of passed SMILES: '+str(passed_num))
            st.dataframe(st.session_state['df'].style.applymap(highlight_result, subset=pd.IndexSlice[:, ['checked']]),use_container_width=True)
            if passed_num == 0:
                st.markdown("<p style='text-align: center; color: red;'><b>The uploaded file contains no suitable SMILES string.</b></p>",unsafe_allow_html=True)
                st.session_state.batch_left_checked = False
                with sidebar_con.container():
                        set_step(0)
                        oab_sidebar(0)
            else:
                st.session_state.batch_left_checked = True
            df = st.session_state['df']
            download_df(df,0)
            choose_score_con = st.empty()
            if st.session_state.batch_left_checked:
                with sidebar_con.container():
                    set_step(2)
                    oab_sidebar(2)
                with choose_score_con.container():
                    with st.form("Choose score to calculate"):
                        st.markdown("<h4>Choose score(s) to calculate</h4>",unsafe_allow_html=True)
                        st.caption('Penalized LogP is always calculated.')
                        st.checkbox('LogP',key='logp_cal')
                        st.checkbox('SA score',key='sa_cal')
                        st.checkbox('Cycle score',key='cycle_cal')
                        calc_batch_butt = st.form_submit_button("Keep passed SMILES and calculate scores")
            else:
                choose_score_con.empty()
    else:
        check_batch_con.empty()
    
    if 'optim_batch_butt' not in locals():
        optim_batch_butt = False
    # if 'calc_batch_butt' in st.session_state:
    if calc_batch_butt and st.session_state.batch_left_checked:
        # if not st.session_state.scores_calculated:
        smiles_list = list(st.session_state['df'][st.session_state['df'].checked != 'invalid'].checked)
        st.session_state.score_df = calc_scores(smiles_list)
        st.session_state.batch_optimized = False
    if st.session_state.scores_calculated:
        calc_batch_con.empty()
        with calc_batch_con.container():
            st.dataframe(st.session_state.score_df,use_container_width=True)
            score_df = st.session_state.score_df
            download_df(score_df,1)
            with sidebar_con.container():
                set_step(3)
                oab_sidebar(3)
            with st.form(":gear: Settings"):
                    st.slider('Choose learning rate: ',0.0,5.0,0.4,key='lr_b')
                    st.slider('Choose similarity cutoff: ',0.0,1.0,0.4,key='sim_cutoff_b')
                    st.slider('Choose number of iterations: ',1,100,80,key='n_iter_b')
                    optim_batch_butt = st.form_submit_button("Optimize")
    else:
        calc_batch_con.empty()
    
    
    optim_batch_con = st.empty()
    ani_con = st.empty()
    if optim_batch_butt and st.session_state.scores_calculated:
        optim_batch_con.empty()
        with sidebar_con.container():
                set_step(4)
                oab_sidebar(4)
        with ani_con.container():
            st.markdown('Operation in progress. Please wait...')
            gen_results = []
            render_animation()
            st.markdown('Generating new SMILES string(s)...')
            model = load_model()
            for canon_smiles in stqdm(list(st.session_state.score_df.SMILES)):
                gen_results.append(optim_single(canon_smiles,model,st.session_state.lr_b,st.session_state.sim_cutoff_b,st.session_state.n_iter_b))
            st.markdown('Checking generated SMILES string(s) ...')
            st.session_state.new_score_df = calc_scores_new(gen_results)
        ani_con.empty()
    if st.session_state.batch_optimized:
        with sidebar_con.container():
                set_step(5)
                oab_sidebar(5)
        with optim_batch_con.container():
            new_score_df = st.session_state.new_score_df
            # new_score_df.style.applymap(highlight_result, subset=pd.IndexSlice[:, ['new_smiles']])
            st.markdown("<h3 style='text-align: center; color: mediumseagreen;'>RESULTS</h3>",unsafe_allow_html=True)
            st.dataframe(new_score_df.style.applymap(highlight_result, subset=pd.IndexSlice[:, ['new_smiles']]),use_container_width=True)
            download_df(new_score_df,3)
    else:
        optim_batch_con.empty()
        

def process_check_single(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if (mol is not None) and (mol_passes_filters_custom(mol) == 'YES'):
        return Chem.MolToSmiles(mol)
    else:
        return 'invalid'
def check_batch(smiles_list):
    check = []
    # check = Parallel(n_jobs=-1,backend='loky')(
    #     delayed(process_check_single)(smi) for smi in stqdm(smiles_list)
    # )
    for smi in stqdm(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if (mol is not None) and (mol_passes_filters_custom(mol) == 'YES'):
                check.append(Chem.MolToSmiles(mol))
            else:
                check.append('invalid')
    st.session_state['df'] = pd.concat([st.session_state['df'],pd.DataFrame({'checked':check})],axis=1)
    st.session_state.checked_batch = True
    # return check

def calc_scores(smiles_list):
    score_df = pd.concat([st.session_state.df[st.session_state.df.checked != 'invalid'].SMILES,pd.DataFrame({'Checked_SMILES':smiles_list})],axis=1)
    scores =[]
    # scores = Parallel(n_jobs=-1,backend='loky')(
    #     delayed(penalized_logp_standard)(Chem.MolFromSmiles(smi)) for smi in stqdm(smiles_list)
    # )
    for smi in stqdm(smiles_list):
        logp,sa,cycle,pen_logp = penalized_logp_standard(Chem.MolFromSmiles(smi))
        scores+=[(logp,sa,cycle,pen_logp)]
    s_df = pd.DataFrame(scores,columns=st.session_state.sc_name)
    for n, checked in zip(st.session_state.sc_name,[st.session_state.logp_cal,st.session_state.sa_cal,st.session_state.cycle_cal,True]):
        if checked:
            score_df = pd.concat([score_df,s_df[n]],axis=1)
    st.session_state.scores_calculated = True
    return score_df

def process_calc_new_score(new_smiles,sim):
    if new_smiles is None:
        return ('invalid',-100.0,-100.0,-100.0,-100.0,-100.0)
    else:
        new_mol = Chem.MolFromSmiles(new_smiles)
        if new_mol is None:
            return ('invalid',-100.0,-100.0,-100.0,-100.0,-100.0)
        else:
            logp,sa,cycle,pen_logp = penalized_logp_standard(new_mol)
            return (new_smiles,sim,logp,sa,cycle,pen_logp)

def calc_scores_new(result):
    new_scores =[]
    # new_scores = Parallel(n_jobs=-1,backend='loky')(
    #     delayed(process_calc_new_score)(new_smiles,sim) for new_smiles,sim in stqdm(result)
    # )
    for new_smiles,sim in stqdm(result):
        if new_smiles is None:
            new_scores+=[('invalid',-100.0,-100.0,-100.0,-100.0,-100.0)]
        else:
            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol is None:
                new_scores+=[('invalid',-100.0,-100.0,-100.0,-100.0,-100.0)]
            else:
                logp,sa,cycle,pen_logp = penalized_logp_standard(new_mol)
                new_scores+=[(new_smiles,sim,logp,sa,cycle,pen_logp)]
    new_col = ['new_smiles','sim']+st.session_state.new_sc_name
    s_df = pd.DataFrame(new_scores,columns=new_col)
    new_score_df = st.session_state.score_df
    for n, checked in zip(new_col,[True, True,st.session_state.logp_cal,st.session_state.sa_cal,st.session_state.cycle_cal,True]):
        if checked:
            new_score_df = pd.concat([new_score_df,s_df[n]],axis=1)
    st.session_state.batch_optimized = True
    return new_score_df

def highlight_result(value):
    if value == 'invalid': color = 'tomato'
    else: color = 'mediumseagreen'
    return 'color: %s' % color

@st.cache_data(experimental_allow_widgets=True)
def download_df(df,id):
    with st.expander(':arrow_down: Download this dataframe'):
        st.markdown("<h4 style='color:tomato;'>Select column(s) to save:</h4>",unsafe_allow_html=True)
        for col in df.columns:
            st.checkbox(col,key=str(id)+'_col_'+str(col),value=True)
        st.text_input('File name (.csv):','dataframe',key=str(id)+'_file_name')
        
        ste.download_button('Download',df_to_file(df[[col for col in df.columns if st.session_state[str(id)+'_col_'+str(col)]]]),st.session_state[str(id)+'_file_name']+'.csv')

def reset_oam_state():
    st.session_state.smiles_selected = False
    st.session_state.checked_single = 'NO'
    st.session_state.smiles_checked = False
    st.session_state.single_optimized = False
    set_step(0)
    
def reset_oab_state():
    st.session_state.smiles_upload_change = True
    st.session_state.smiles_uploaded = False
    st.session_state.checked_batch = False
    st.session_state.batch_left_checked = False
    st.session_state.scores_calculated = False
    st.session_state.batch_optimized = False
    set_step(0)

def rerun():
    st.experimental_rerun()
def render_view():
    # render_sidebar(st.session_state.current_view,st.session_state.current_step)
    form_header()
    form_body()


render_view()
