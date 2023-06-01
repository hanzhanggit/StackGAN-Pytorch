import argparse
import csv
import os.path
import pathlib
import pickle
import shutil
from collections import defaultdict
from datetime import datetime
from pprint import pprint
from easydict import EasyDict as edict

from voc_tools.constants import VOC_IMAGES
from voc_tools.reader import list_dir, from_file
from voc_tools.utils import Dataset, VOCDataset
import sqlite3


def generate_class_id_pickle(dataset_path, classes):
    """
    Generate and save CLASS_IDS into a pickle file

    Args:
        dataset_path: VOC dataset path
        classes: a list of classes to add to dataset
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))
    # categorical encoding of the classes
    unique = list(set(classes))
    class_id = list(map(lambda x: unique.index(x) + 1, classes))
    # save as pickle file
    pickle_class_info_path = str(dataset_path / "train" / "class_info.pickle")
    with open(pickle_class_info_path, 'wb') as fpp:
        pickle.dump(class_id, fpp)
        print("'{}' is created with {} entries".format(pickle_class_info_path, len(class_id)))


def get_embedding_model(fasttext_model_path=None, emb_dim=300):
    """
    Args:
        fasttext_model_path: Pretrained fasttext  model (*.bin) file path See: https://fasttext.cc/docs/en/crawl-vectors.html
        emb_dim: Final embedding dimension
    """
    import fasttext
    import fasttext.util
    model = None
    model_name = ""
    if os.path.isfile(fasttext_model_path):
        model_name = pathlib.Path(fasttext_model_path).name
        print("Loading fasttext model:{}...".format(fasttext_model_path), end="")
        model = fasttext.load_model(fasttext_model_path)
        print("Loaded")
        if emb_dim != model.get_dimension():
            fasttext.util.reduce_model(model, emb_dim)
    assert model is not None, "A fasttext  model has to be initialised"
    return model, model_name, emb_dim


def generate_text_embedding_pickle(dataset_path, embeddings, model_name, emb_dim, mode="train"):
    """
    Generate and save caption embedding into a pickle file.
    Args:
        dataset_path: VOC dataset path
        embeddings: Prepared caption embeddings
        model_name: Model name which is used to prepare the embeddings
        emb_dim: Final embedding dimension
        mode: Generate embeddings for training or testing
    """
    assert mode in ("train", "test"), mode
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))
    # save as pickle file
    pickle_path = str(dataset_path / mode / "embeddings_{}_{}D.pickle".format(model_name, emb_dim))
    with open(pickle_path, 'wb') as fpp:
        pickle.dump(embeddings, fpp)
        print("'{}' is created with {} entries".format(pickle_path, len(embeddings)))


def generate_filename_pickle(dataset_path, filenames):
    """
    Generate a list of 'filenames' per 'caption', then save it as pickle format.
    Note: This is not a unique filename list. If, for a certain image, we have multiple
    captions, then multiple entries with same 'filename' will be created.

    Args:
        dataset_path: the path to the dataset
    """
    dataset_path = pathlib.Path(dataset_path)
    assert os.path.exists(str(dataset_path))

    pickle_filenames_path = str(dataset_path / "train" / "filenames.pickle")

    # save as pickle file
    with open(pickle_filenames_path, 'wb') as fpp:
        pickle.dump(filenames, fpp)
        print("'{}' is created with {} entries".format(pickle_filenames_path, len(filenames)))


class DatasetWrap:
    def __init__(self, dataset_path, bulk=False, class_ids=False, fasttext_model_path=None,
                 embedding_dimension=300, test_captions=None) -> None:
        dataset_path = pathlib.Path(dataset_path)
        assert os.path.exists(str(dataset_path))
        self.dataset_path = pathlib.Path(dataset_path)
        self.is_bulk = bulk
        self.class_ids = class_ids
        self.embedding_dim = embedding_dimension
        self.test_captions = test_captions
        if fasttext_model_path is not None:
            # loading embedding
            self.emb_model, self.emb_model_name, _ = get_embedding_model(fasttext_model_path, embedding_dimension)
        else:
            self.emb_model, self.emb_model_name = None, None
        self.voc_data = VOCDataset(dataset_path, caption_support=True)

    def _prepare_classes(self):
        """
            Each image typically contains one or many captions. In two ways we can save the 'class_ids'
        into the pickle file, 'class_ids' per caption and 'class_ids' per image. 'bulk' parameter
        controls this action.
        """
        dataset_path = self.dataset_path
        if self.is_bulk:
            # 'class_ids per image'
            self.classes = [annotations[0].class_name for annotations, jpeg in self.voc_data.train.fetch(bulk=True)]
        else:
            # 'class_ids per caption'
            self.classes = list(
                map(lambda caption:
                    list(from_file(str(dataset_path / "train" / Dataset.CAPTION_DIR / caption.filename)))[
                        0].class_name,
                    self.voc_data.train.caption.fetch(bulk=False)))
        print("Class_id is prepared")

    def _prepare_filenames(self):
        """
            Each image typically contains one or many captions. In two ways we can save the filenames
        into the pickle file, filename per caption and filename per image. 'bulk' parameter
        controls this action.
        """
        dataset_path = self.dataset_path

        if self.is_bulk:

            mylist = [os.path.basename(file) for file in list_dir(str(dataset_path / "train"), dir_flag=VOC_IMAGES)]
        else:
            # read the filenames from captions
            mylist = [caption.filename.replace(".txt", ".jpg") for caption in
                      self.voc_data.train.caption.fetch(bulk=False)]
        self.filenames = list(map(lambda x: os.path.join("train", Dataset.IMAGE_DIR, x), mylist))  # fill path names
        print("Filenames is prepared")

    @staticmethod
    def clean(text):
        import re
        # remove punctuation, space, linefeed etc.
        text = re.sub(r'[^\w\s\']', ' ', text)
        text = re.sub(r'[ \n]+', ' ', text)
        return text.strip().lower()

    def create_fasttext_data(self, text):
        with open("./temp_text_data.txt", "w", encoding="utf-8") as fp:
            fp.write(self.clean(text))
        return "./temp_text_data.txt"

    def train_fasttext_model(self, caption_data, fasttext_cfg):
        assert fasttext_cfg, "'fasttext_cfg' is required to train a fasttext model"
        keys = ("epoch", "lr", "algorithm")
        assert all(map(lambda x: x in keys,
                       fasttext_cfg.keys())), "The following keys are required:{} in 'fasttext_cfg'".format(keys)
        # train embedding model
        import fasttext
        data_path = self.create_fasttext_data(caption_data)
        model = fasttext.train_unsupervised(data_path, fasttext_cfg.algorithm, dim=self.embedding_dim, thread=4,
                                            epoch=fasttext_cfg.epoch, lr=fasttext_cfg.lr)
        self.emb_model = model
        self.emb_model_name = "fasttext_{}_{}".format(fasttext_cfg.algorithm, self.embedding_dim)
        # remove temporary files
        os.remove(data_path)
        print("Fasttext model is trained")

    def _prepare_embeddings(self, model, fasttext_cfg=None):
        """
            Each image typically contains one or many captions. In two ways we can save the caption
        embeddings into the pickle file. In as single instance or in bulk. Let, say we have 3 captions per image, and we
        have 2 images. If we want to store the embeddings as bulk, then it should look like:
        [[emb11, emb12, emb13], [emb21, emb22, emb23]]. If we want to store the embeddings as single instance, then it will be:
        [[emb11], [emb12], [emb13], [emb21], [emb22], [emb23]].
        """
        dataset_path = self.dataset_path
        mydata = VOCDataset(dataset_path, caption_support=True)
        if model is None:
            # get caption data
            caption_data = "".join(
                set(list(map(lambda caption: caption.captions.strip().strip(".").strip(),
                             mydata.train.caption.fetch(bulk=False)))))
            # train model
            self.train_fasttext_model(caption_data, fasttext_cfg)
            model = self.emb_model

        if self.is_bulk:
            self.embeddings = [list(map(lambda cap: model.get_word_vector(cap.captions), caption_list)) for caption_list
                               in mydata.train.caption.fetch(bulk=True)]
        else:
            self.embeddings = [[model.get_word_vector(caption.captions)] for caption in
                               mydata.train.caption.fetch(bulk=False)]
        print("Text embeddings is prepared for training")
        if self.test_captions is not None:
            self.test_embeddings = list(map(lambda cap: model.get_word_vector(cap), self.test_captions))
            print("Text embeddings is prepared for testing")

    def prepare_dataset(self, fasttext_cfg=None):
        if self.class_ids:
            self._prepare_classes()
        self._prepare_filenames()
        self._prepare_embeddings(self.emb_model, fasttext_cfg)
        generate_filename_pickle(str(self.dataset_path), self.filenames)
        if self.class_ids:
            generate_class_id_pickle(str(self.dataset_path), self.classes)
        generate_text_embedding_pickle(str(self.dataset_path), self.embeddings, self.emb_model_name, self.embedding_dim)
        if self.test_captions is not None:
            generate_text_embedding_pickle(str(self.dataset_path), self.test_embeddings, self.emb_model_name,
                                           self.embedding_dim, mode="test")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--data_dir', dest='data_dir', help='Path to dataset directory.',
                        default='my_data', type=str, )
    parser.add_argument('--test_data_file', dest='test_data_file', help='A text file contains unseen captions to test',
                        default='', type=str, )
    parser.add_argument('--bulk', dest='bulk', default=False, action="store_true")
    parser.add_argument('--class_id', dest='class_id', default=False, action="store_true")
    parser.add_argument('--fasttext_train_lr', dest='fasttext_train_lr', type=float, default=None)
    parser.add_argument('--fasttext_train_algo', dest='fasttext_train_algo', type=str, default=None,
                        choices=['skipgram', 'cbow'])
    parser.add_argument('--fasttext_train_epoch', dest='fasttext_train_epoch', type=int, default=None)
    parser.add_argument('--fasttext_model', dest='fasttext_model', type=str, default=None)
    parser.add_argument('--emb_dim', dest='emb_dim', type=int, default=300)
    parser.add_argument('--sqlite', dest='sqlite',
                        help='Path to SQLite3 database file.',
                        default='', type=str, )
    parser.add_argument('--clean', dest='clean', default=False, action="store_true",
                        help="Clean before generating new data while using sqlite")
    parser.add_argument('--copy_images', dest='copy_images', default=False, action="store_true",
                        help="Copy images while generating new data using sqlite, --dataroot option is required")
    parser.add_argument('--dataroot', dest="dataroot", type=str,
                        help="This is a path to a image dataset to copy images while creating dataset using sqlite."
                             "The dataset format is default to PascalVOC format.")
    _args = parser.parse_args()
    pprint(_args)
    return _args


def generate_dataset(args):
    # creating object to generate compatible pickle files for StackGAN
    # prepare fasttext training configuration
    fasttext_cfg = edict()
    fasttext_cfg.epoch = args.fasttext_train_epoch
    fasttext_cfg.lr = args.fasttext_train_lr
    fasttext_cfg.algorithm = args.fasttext_train_algo
    # read the test captions
    test_captions = None
    if os.path.isfile(args.test_data_file):
        with open(args.test_data_file) as fp:
            test_captions = fp.readlines()
    # initialize dataset
    vdw = DatasetWrap(args.data_dir, args.bulk, args.class_id,
                      fasttext_model_path=args.fasttext_model,
                      embedding_dimension=args.emb_dim, test_captions=test_captions)
    vdw.prepare_dataset(fasttext_cfg)


def get_openai_api_key(path):
    with open(path, "r") as fp:
        for key, nickname in csv.reader(fp):
            yield key, nickname


class OpenAICredentialManager:
    """
    This singleton class read a working api key form a list of api-keys available as a simple csv file.
    It yields an endless flow of valid openai api key. Background: The usage limit for an OpenAI API
    key is limited as described here: https://platform.openai.com/docs/guides/rate-limits/overview.
    This is useful, when users have multiple OpenAI account to leverage the full potential of openai seamlessly.

    Save the file:
    openapi-key.csv
        sk-x12x12x12x12xcc125221c223444XfYtyTYTgtYTYTyyTybb,user1
        sk-x12x12x12x12xcc125221c223444XfYtyTYTgtYTYTyyTybb,user2

    Example:
        cm = OpenAICredentialManager("openapi-key.csv")
        key, nickname = next(cm) # sk-x12x12x12x12xcc125221c223444XfYtyTYTgtYTYTyyTybb, user1

    """
    instance = None

    def __init__(self, path):
        if OpenAICredentialManager.instance is not None:
            raise Exception("'{}' is a singleton class!".format(self.__class__.__qualname__))
        OpenAICredentialManager.instance = self
        self.path = path
        self.nickname = None
        self.key = None
        self.keygen = get_openai_api_key(path)

    def __next__(self):
        try:
            self.key, self.nickname = next(self.keygen)
        except StopIteration:
            self.keygen = get_openai_api_key(self.path)
            self.key, self.nickname = next(self.keygen)
        finally:
            print("Using openai API Key for: '{}'".format(self.nickname))
        return self.key, self.nickname


class OpenAITextPreprocessor:
    """
    OpenAI API access is limited as describer here: https://platform.openai.com/docs/guides/rate-limits/overview.
    It limits user by Request Per Minute(RPM) and Token Per Minute(TPM). And this also varies as the choice of
    algorithm. Thus, to make the most out of the algorithm we can preprocess the texts in bulk (as documents)
    before sending out to openapi. This way we can optimize the RPM and TPM
    """

    def __init__(self, model="ada", rpm=3, tpm=150000):
        """
        Args:
            model: which model to use. Default choice is 'ada'. As per openai,
                   you can send approximately 200x more tokens per minute to an 'ada' model versus a 'davinci' model.
            rpm: How much RPM you are allowed
            tpm: How much TMP you are allowed
        """
        self.start_time = datetime.now()


def generate_dataset_with_langchain():
    from langchain.embeddings import OpenAIEmbeddings
    from openai.error import RateLimitError

    cm = OpenAICredentialManager("./data/openai.apikey")
    key, nickname = next(cm)

    captions = ["Security discovered a concealed knife in the passenger's bag."]
    embedding_map = []
    ctr = 0
    while ctr < len(captions):
        model = OpenAIEmbeddings(openai_api_key=key,
                                 model="ada",
                                 max_retries=1)
        try:
            embeddings = model.embed_query(captions[ctr])
            embedding_map.append([captions[ctr], embeddings])
            ctr += 1
        except RateLimitError:
            key, nickname = next(cm)
    # A quick brown fox jump over the lazy dog
    arr1 = [-0.008165569044649601, 0.006291043478995562, -0.012001951225101948, -0.003452744334936142,
            -0.013524028472602367, 0.01843958906829357, -0.020597944036126137, -0.00899522565305233,
            -0.007635336834937334, -0.030216971412301064, 0.020173758268356323, 0.01786569133400917,
            0.020760131999850273, 0.0018838822143152356, 0.006917964667081833, -0.009038891643285751,
            0.030591251328587532, -0.005526886321604252, 0.011771144345402718, -0.00923850852996111,
            -0.018489493057131767, 0.019662240520119667, -0.002724455436691642, -0.002885084366425872,
            0.0071674855425953865, 0.015083533711731434, 0.006004094611853361, -0.006000975612550974,
            -0.007161247543990612, 0.0005236038123257458, 0.01847701705992222, -0.0011423374526202679,
            -0.0338599756360054, -0.000905292690731585, -0.009020177647471428, -0.01576971635222435,
            -0.008365185000002384, -0.0025248387828469276, 0.022743823006749153, -0.04478898644447327,
            0.014871440827846527, 0.02209506928920746, -0.020048998296260834, -0.013386791571974754,
            0.002119367476552725, 0.010904059745371342, 0.014746679924428463, -0.013074890710413456,
            -0.02486474998295307, 0.011278340592980385, -0.0007470029522664845, 0.012026903219521046,
            -0.008333995006978512, -0.007354625966399908, -0.010785537771880627, -0.0009949642699211836,
            -0.00830280501395464, 0.004572468809783459, 0.013212127611041069, -0.005848144181072712,
            0.004048475064337254, 0.010311448015272617, -0.005096462555229664, -0.002464018063619733,
            -0.0013076449977234006, -0.011328245513141155, 0.010267782025039196, -0.0007466130773536861,
            -0.015981808304786682, 0.0012444850290194154, 0.01610656827688217, -0.0006600605556741357,
            0.0024780535604804754, 0.0058949291706085205, 0.0315394327044487, -0.006917964667081833,
            -0.035107579082250595, 0.014734203927218914, 0.021072033792734146, 0.010411255992949009,
            0.028919462114572525, -0.03695403411984444, -0.0062380204908549786, 0.017229411751031876,
            0.022232305258512497, 0.0058949291706085205, -0.024228472262620926, 0.030067259445786476,
            -0.01484648883342743, 0.00378024042584002, 0.00023918910301290452, 0.03036668337881565,
            0.005536243319511414,
            -0.0033217458985745907, -0.016992367804050446, 0.0009707919089123607, -0.006843108218163252,
            0.01871406100690365, -0.012363756075501442, -0.03303655609488487, 0.006053999066352844,
            -0.005214984994381666, -0.026249589398503304, 0.0002329510753042996, 0.008421327918767929,
            -0.026249589398503304, -0.012020665220916271, 0.018165115267038345, 0.019138246774673462,
            -0.007691479288041592, 0.012132950127124786, 0.013286983594298363, 0.01649332605302334,
            -0.036280326545238495, -0.0002345105749554932, 0.0002649209345690906, 0.014921344816684723,
            -0.008689562790095806, -0.009662693366408348, -0.009088795632123947, 0.012338804081082344,
            0.028595086187124252, 0.02350486069917679, -0.004142045509070158, -0.015557622537016869,
            -0.009893500246107578, 0.0029537025839090347, -0.001486208406277001, -0.010361352004110813,
            -0.002042951760813594, 0.0315643846988678, 0.023642096668481827, 0.024739988148212433,
            -0.009275936521589756,
            -0.0396987609565258, 0.025550931692123413, 0.007697717286646366, 0.003696027211844921,
            -0.023454956710338593,
            -0.019587384536862373, 0.006724586244672537, 0.021783167496323586, -0.007073915097862482,
            0.001193801173940301, 0.019362814724445343, 0.02548855170607567, 0.010305210016667843,
            -0.004120212513953447,
            0.0025466717779636383, -0.014609443955123425, 0.014921344816684723, 0.0027587644290179014,
            0.03495786711573601, 0.0016000522300601006, 0.00869580078870058, 0.014310019090771675,
            0.004666039254516363,
            0.01670541800558567, -0.026000069454312325, -0.013998118229210377, 0.025700643658638,
            0.022606585174798965,
            -0.004251210950314999, -0.011970761232078075, -0.0024016378447413445, 0.022631539031863213,
            0.027297576889395714, -0.00461301626637578, 0.013249555602669716, -0.008296567015349865,
            0.0025763025041669607, 0.03128990903496742, -0.013511552475392818, 0.0339098796248436,
            -0.00505591556429863,
            0.01157152745872736, 0.009949642233550549, -0.009762502275407314, -0.030117163434624672,
            0.0004120992380194366, -0.003200104460120201, 0.015570099465548992, 0.035107579082250595,
            0.020760131999850273, 0.004447708372026682, 0.027971284464001656, 0.010473635978996754,
            -0.03191371262073517,
            -0.0014971249038353562, 0.00020254073024261743, 0.005062153562903404, 0.005305436439812183,
            -0.011989475227892399, -0.013923261314630508, -0.6874797344207764, -0.021932879462838173,
            0.009600313380360603, -0.007597908843308687, 0.020523088052868843, 0.03178895264863968,
            -0.01033016201108694,
            -0.012326328083872795, 0.006044641602784395, 0.031464576721191406, -0.030192019417881966,
            0.0054208398796617985, 0.0037958354223519564, -0.0048375846818089485, -0.006749538239091635,
            -0.0038582156412303448, 0.02896936610341072, -0.020061474293470383, -0.009076319634914398,
            0.01649332605302334, 0.0030176423024386168, 0.009918452240526676, -0.011584004387259483,
            -0.0010238151298835874, -0.0016671109478920698, 0.00846499390900135, 0.0015602848725393414,
            0.0022940319031476974, 7.665942393941805e-05, 0.013124794699251652, 0.002007083036005497,
            0.013037462718784809, -0.014821536839008331, 0.005458267871290445, 0.05224965885281563,
            -0.0004378310695756227, 0.0002906527661252767, 0.018901202827692032, 0.031364765018224716,
            0.05195023491978645, -0.033835023641586304, -0.00484694167971611, 0.014347447082400322,
            0.0019213103223592043, -0.019537480548024178, -0.0015150592662394047, 0.02346743270754814,
            0.014160306192934513, 0.00015058970893733203, -0.02033594623208046, 0.004778323695063591,
            0.020972223952412605, 0.020398326218128204, 0.004759609699249268, -0.018077783286571503,
            0.022369541227817535, 0.01775340549647808, 0.008284091018140316, 0.0068867746740579605,
            -0.0010604634881019592, -0.007822477258741856, 0.003312388900667429, -0.013573932461440563,
            0.015270673669874668, -0.023854190483689308, -0.0026059329975396395, -0.009556647390127182,
            0.015333054587244987, 0.017578741535544395, -0.016293710097670555, 0.015707336366176605,
            0.02185802347958088,
            -0.017453981563448906, 0.00747314840555191, 0.014347447082400322, 0.00674330024048686,
            0.0012873715022578835,
            -0.007173723541200161, -0.016393518075346947, 0.010005785152316093, -0.01426011510193348,
            -0.002663634717464447, -0.033360932022333145, 0.004154521506279707, 0.029717929661273956,
            0.015557622537016869, -0.033111412078142166, -0.010199163109064102, -0.003012963803485036,
            -0.021870499476790428, 0.018888724967837334, 0.02909412793815136, -0.03538205102086067,
            0.003115891246125102,
            -0.007735145278275013, 0.0040952605195343494, -0.012719323858618736, 0.041170936077833176,
            0.04241853952407837, -0.02525150589644909, 0.009076319634914398, -0.0034059591125696898,
            -0.008390137925744057, -0.009643979370594025, 0.011758668348193169, 0.025750547647476196,
            0.019188150763511658, 0.00555183831602335, 0.024852273985743523, -0.0096502173691988,
            -0.00805952213704586,
            -0.018302351236343384, 0.02115936577320099, 0.009257222525775433, -0.0065686353482306,
            -0.02276877500116825,
            0.012114236131310463, -0.005604861304163933, 0.012519706971943378, -0.022132497280836105,
            0.011484195478260517, -0.012594562955200672, 0.014447255060076714, 0.00645635137334466,
            0.00935079250484705,
            0.005074629560112953, -0.03410949558019638, -0.027522146701812744, -0.011671336367726326,
            -0.010691966861486435, -0.00645635137334466, 0.022282209247350693, 0.029393551871180534,
            -0.0047845616936683655, -0.00280399015173316, -0.00823418702930212, 0.010891583748161793,
            0.009151175618171692, 0.017728453502058983, -0.010536016896367073, -0.023791810497641563,
            0.01004321314394474, -0.005083986558020115, -0.018027879297733307, 0.0026402422226965427,
            -0.020298518240451813, -0.026324445381760597, -0.004126450512558222, 0.0065436833538115025,
            0.02979278564453125, -0.019537480548024178, -0.020273566246032715, -0.009400696493685246,
            -0.013162222690880299, 0.01542038656771183, -0.009450601413846016, -0.032512564212083817,
            -0.005720264744013548, -0.010074403136968613, -0.033136364072561264, 0.037577833980321884,
            0.0059074051678180695, -0.0008873583865351975, 7.826766523066908e-05, 0.007142533548176289,
            -0.030092211440205574, 0.00695539265871048, 0.024727512151002884, -0.025525979697704315,
            -0.019699668511748314, -0.011222198605537415, -0.024153614416718483, -0.010866631753742695,
            -0.001426167436875403, -0.01740407757461071, 0.009082557633519173, -0.024116186425089836,
            -0.02852023020386696, -0.00528672244399786, -0.010149259120225906, -0.005941714625805616,
            0.01419773418456316, 0.007136295549571514, -0.01892615482211113, 0.013773549348115921,
            -0.0011743073118850589, 0.02311810292303562, 0.021059557795524597, -0.019188150763511658,
            -0.011783620342612267, 0.024340756237506866, 0.03071601316332817, 0.009662693366408348,
            -0.004251210950314999, -0.0005918322131037712, -0.018888724967837334, 0.0036430039908736944,
            0.0251017939299345, 0.012020665220916271, 0.01408545020967722, 0.012457326985895634,
            -0.007722669281065464,
            0.005049677565693855, -0.03341083601117134, 0.029817737638950348, -0.03201352059841156,
            0.007853668183088303,
            -0.00411709351465106, 0.005190032999962568, -0.0016827060608193278, -0.0192754827439785,
            0.0032905556727200747, 0.01542038656771183, -0.029618121683597565, 0.01973709650337696,
            0.022494301199913025,
            -0.012500992976129055, 0.023654572665691376, 0.006500017363578081, -0.01775340549647808,
            0.00322817568667233,
            -0.012875273823738098, 0.014447255060076714, -0.009132461622357368, 0.01881386898458004,
            0.01717950776219368,
            0.010448683984577656, 0.016979891806840897, -0.003025439800694585, -0.006041522603482008,
            -0.026723679155111313, 0.0033560548909008503, 0.02031099423766136, 0.013262031599879265,
            0.010361352004110813, -0.010255306027829647, 0.00835894700139761, -0.010336400009691715,
            0.015345530584454536, 0.0045163268223404884, 0.018514445051550865, -0.0029427860863506794,
            0.005916762165725231, -0.017204459756612778, 0.01046739798039198, 0.012987558729946613,
            0.023991426452994347,
            -0.0011205044575035572, -0.016430946066975594, 0.0027135389391332865, -0.0021443194709718227,
            -0.007142533548176289, -0.024166090413928032, 0.03201352059841156, 0.011016343720257282,
            -0.01660561002790928, 0.012787941843271255, 0.0013481922214850783, 0.022257257252931595,
            0.025089317932724953, 0.010030737146735191, -0.00035303295589983463, 0.010810489766299725,
            0.009275936521589756, 0.02266896702349186, -0.0022612824104726315, -0.011577766388654709,
            0.010810489766299725, 0.01192085724323988, -0.03106534108519554, -0.02431580424308777,
            -0.0021365219727158546, -0.0037584074307233095, -0.010479873977601528, 0.024240948259830475,
            0.01786569133400917, -0.035357099026441574, 0.02009890228509903, 0.019549956545233727,
            0.02243192121386528,
            -0.025525979697704315, -0.029168983921408653, -0.0007181520923040807, 0.0065312073566019535,
            -0.0027306934352964163, 0.0051401290111243725, -0.021171841770410538, 0.0003928003425244242,
            -6.038208812242374e-05, 0.02314305491745472, -0.004460184834897518, 0.0007781930617056787,
            -0.0035556715447455645, -0.0011368792038410902, -0.006687157787382603, -0.01658065803349018,
            0.019100818783044815, -0.004232496954500675, -0.006203711498528719, -0.02500198595225811,
            0.013274507597088814, -0.0025934570003300905, -0.004332305397838354, -0.009905976243317127,
            0.025263981893658638, 0.0007544105756096542, -0.010660776868462563, -0.0032905556727200747,
            0.003948667086660862, 0.018302351236343384, -0.01588200032711029, -0.008165569044649601,
            -0.012837845832109451, 0.012975082732737064, -0.002292472403496504, -0.007335911970585585,
            -0.026224637404084206, -0.012001951225101948, 0.025700643658638, 0.014135354198515415,
            -0.00605087960138917,
            -0.013386791571974754, -0.006911726668477058, 0.019075866788625717, 0.09032653272151947,
            0.023978950455784798, -0.012476040981709957, 0.016742845997214317, 0.015345530584454536,
            -0.019724620506167412, -0.022232305258512497, -0.009537933394312859, 0.023642096668481827,
            -0.008602229878306389, -0.0015610646223649383, -0.0065935878083109856, 0.002842977875843644,
            -0.00402976106852293, 0.02440313622355461, -0.012232758104801178, 0.010579682886600494,
            -0.013049938715994358, 0.0038426206447184086, -0.0036804319825023413, -0.0007544105756096542,
            0.0017762762727215886, 0.0009926250204443932, 0.005427077878266573, -0.016044188290834427,
            0.006231782492250204, -0.008751942776143551, 0.019001010805368423, 0.01941271871328354,
            -0.03141467273235321,
            0.0016172068426385522, 0.01128457859158516, 0.012338804081082344, 0.004110855516046286,
            0.021870499476790428,
            0.010748108848929405, -0.0033997211139649153, 0.023317720741033554, 0.032637324184179306,
            -0.024615228176116943, 0.01600676029920578, 0.012688133865594864, 0.0024811725597828627,
            0.01034263800829649,
            0.023891618475317955, -0.0005466065485961735, -0.028220804408192635, 0.007173723541200161,
            0.008945321664214134, -0.029717929661273956, 0.031389717012643814, -0.005941714625805616,
            -0.010486111976206303, 0.00922603253275156, 0.008221711032092571, 0.025288935750722885,
            -0.008976511657238007, 0.0009364828001707792, -0.011365673504769802, -0.008159331046044827,
            -0.011234674602746964, -0.017578741535544395, -0.009750026278197765, -0.00986854825168848,
            0.005068391561508179, -0.026199685409665108, -0.016655514016747475, -0.011178532615303993,
            -0.006172521039843559, -0.016418470069766045, -0.01868910901248455, -0.02464018017053604,
            -0.017341697588562965, -0.001334156608209014, 0.010348876006901264, 0.026698727160692215,
            0.0039954520761966705, -0.010186687111854553, 0.006184997037053108, 0.0017965498846024275,
            -0.012644467875361443, -0.005632932297885418, 0.004304233938455582, -0.029942497611045837,
            0.007922286167740822, 0.004815751686692238, 0.015931904315948486, -0.009762502275407314,
            0.002827382879331708, 0.017341697588562965, 0.015557622537016869, 0.004528802819550037,
            -0.002747847931459546, 0.00587309617549181, 0.021309077739715576, -0.008882940746843815,
            0.01424763910472393,
            0.03705384209752083, 0.013037462718784809, 0.0021723906975239515, 0.033810071647167206,
            -0.02057299204170704,
            -0.002679229713976383, -0.0004998213844373822, 0.008714514784514904, -0.005327269434928894,
            0.030441539362072945, -0.0032936749048531055, -0.026723679155111313, -0.019936712458729744,
            0.016430946066975594, -0.0075230528600513935, 0.01634361408650875, 0.003618051763623953,
            0.01034263800829649,
            0.017578741535544395, 0.019699668511748314, 0.013212127611041069, -0.007498100399971008,
            -0.013798501342535019, -0.002495208289474249, -0.02150869369506836, 0.0002664804342202842,
            0.024390660226345062, -0.025975117459893227, 0.01952500455081463, -0.007766335271298885,
            -0.01833977922797203, 0.001069820486009121, 0.0057015507481992245, 0.009787454269826412,
            0.0034153161104768515, -0.002916274592280388, -0.010492349974811077, -0.042693011462688446,
            -0.011945809237658978, -0.0026371232233941555, 0.014285067096352577, -0.018763964995741844,
            -0.00592611962929368, -0.030591251328587532, 0.002339257625862956, 0.011902143247425556,
            -0.0408964604139328,
            0.017491409555077553, -0.028120996430516243, -0.0036024567671120167, -0.0056017423048615456,
            -0.00830280501395464, 0.008440041914582253, -0.006256734486669302, -0.007398292422294617,
            -0.02417856641113758, -0.00870827678591013, -0.00082497822586447, -0.03645499050617218,
            0.01069820486009121,
            -0.010380065999925137, 0.03890029713511467, 0.019774524495005608, 0.033485691994428635,
            -0.0071799615398049355, 0.0036430039908736944, 0.024253424257040024, -0.0262994933873415,
            -0.0035151245538145304, 0.006805680226534605, -0.018414637073874474, -0.013648788444697857,
            -0.006768252234905958, 0.033236172050237656, 0.008415089920163155, 0.03333598002791405,
            0.01858930103480816,
            -0.01847701705992222, 0.006805680226534605, -0.007604146841913462, 0.002033594762906432,
            -0.007697717286646366, -0.01348660048097372, 0.020123854279518127, -0.014160306192934513,
            0.006662205792963505, -0.0033311028964817524, -0.012525944970548153, -0.0068992506712675095,
            0.00840885192155838, 0.029019271954894066, 0.030815821141004562, 0.009843596257269382,
            0.023978950455784798,
            0.00583878718316555, -0.004943631123751402, 0.003000487806275487, -0.01091653574258089,
            -0.023667048662900925, -0.0001323629985563457, 0.011577766388654709, 0.01162143237888813,
            0.02934364788234234, 0.0013910785783082247, 0.01646837405860424, -0.013436696492135525,
            0.03049144335091114,
            0.001966536045074463, 0.01045492198318243, -0.003067546524107456, -0.017266839742660522,
            -0.001685825060121715, -0.025064365938305855, -0.007785049732774496, -0.04007304459810257,
            -0.014871440827846527, -0.02782157063484192, 0.007866144180297852, 0.0245278961956501,
            -0.007972190156579018,
            0.01449715904891491, -0.009718836285173893, -0.037103746086359024, -0.00484694167971611,
            0.002347055124118924, 0.03987342491745949, -0.011883429251611233, 0.0192754827439785,
            0.01613152027130127,
            0.00012836676614824682, 0.003499529557302594, 0.020061474293470383, 0.006724586244672537,
            0.004076546523720026, 0.0021817476954311132, 0.015956856310367584, -0.006359661929309368,
            -0.007747621275484562, -0.03141467273235321, -0.0013224603608250618, -0.025675691664218903,
            -0.0065935878083109856, 0.028719846159219742, -0.0027260149363428354, -0.0008787811384536326,
            -0.011783620342612267, -0.012850321829319, 0.0002271029370604083, 0.011596480384469032,
            -0.014821536839008331, -0.00587309617549181, -0.032762084156274796, -0.0018729656003415585,
            -0.04264310747385025, 0.010442445985972881, -0.020398326218128204, 0.0012195330346003175,
            -0.027871474623680115, -0.004032880067825317, 0.010798013769090176, 0.013536504469811916,
            -0.023766858503222466, -0.005068391561508179, -0.014696775935590267, 0.01716703176498413,
            -0.019450148567557335, -0.000755190325435251, 0.013998118229210377, 0.003225056454539299,
            -0.018264923244714737, -0.017341697588562965, -0.0061226170510053635, 0.026324445381760597,
            -0.01268189586699009, -0.01739160157740116, -0.005180676002055407, 0.01465934794396162,
            0.01636856608092785,
            -0.016181424260139465, 0.0030176423024386168, -0.02230716124176979, -0.004672277253121138,
            0.013224603608250618, 0.016568182036280632, -0.007859906181693077, -0.00583878718316555,
            0.00865213479846716,
            -0.012301376089453697, 0.00257162400521338, -0.0017435266636312008, -0.014509635977447033,
            0.008957797661423683, -0.026099877431988716, -0.0014947856543585658, -0.01600676029920578,
            -0.00514324801042676, 0.0027743596583604813, 0.02093479596078396, -0.000456545123597607,
            0.0026745512150228024, 0.016655514016747475, -0.03513253107666969, 0.016393518075346947,
            -0.022020211443305016, 0.012295138090848923, -0.01740407757461071, 0.01542038656771183,
            0.016381042078137398,
            0.0074606724083423615, 0.014584491960704327, -0.014160306192934513, 0.0005368596175685525,
            -0.030167067423462868, -0.00807199813425541, -0.005274246446788311, 0.014334971085190773,
            -0.02501446194946766, 0.005738978739827871, -0.0010004225187003613, 0.006284805480390787,
            -0.022469349205493927, -0.0003391923673916608, 0.03236284852027893, -0.00011540338164195418,
            0.020286042243242264, 0.03518243506550789, -0.018951106816530228, 0.006843108218163252,
            0.0007661068812012672, 0.00794723816215992, 0.016268758103251457, -0.01589447632431984,
            -0.004241853952407837, -0.006393970921635628, 0.02979278564453125, 0.005904286168515682,
            -0.01705474779009819, -0.015981808304786682, -0.005190032999962568, -0.017828263342380524,
            0.017666073516011238, -0.02276877500116825, -0.004584944806993008, 0.01787816733121872,
            0.013686216436326504,
            -0.024016378447413445, 0.009781216271221638, -0.015333054587244987, -0.015707336366176605,
            -0.011796096339821815, -0.010211639106273651, -0.02991754561662674, -0.02279372699558735,
            -0.002796192653477192, 0.010161735117435455, 0.015382958576083183, -0.008284091018140316,
            -0.02595016546547413, -0.006624777801334858, -0.009332078509032726, -0.011833524331450462,
            0.027646906673908234, 0.0028445373754948378, 0.0077289072796702385, -0.017715977504849434,
            0.007286007981747389, 0.018077783286571503, -0.013561456464231014, 0.03049144335091114,
            -0.017840739339590073, -0.011365673504769802, -0.01016797311604023, -0.005046558566391468,
            -0.003908119630068541, -0.006562397349625826, 0.004812632687389851, -0.01449715904891491,
            0.006762014236301184, 0.027197768911719322, 0.007379577960819006, 0.011240912601351738,
            -0.007984666153788567, 0.005891810171306133, -0.007310959976166487, -0.021084509789943695,
            -0.004282400943338871, -0.0093383165076375, 0.010947725735604763, -0.026449207216501236,
            -0.01821501925587654, 0.013436696492135525, -0.017104651778936386, -0.01833977922797203,
            -0.002936548087745905, 0.014709251932799816, -0.002520160283893347, 0.0023252221290022135,
            -0.018177591264247894, 0.0014246079372242093, -0.018165115267038345, -0.029518311843276024,
            0.0034153161104768515, -0.017079699784517288, 0.025076841935515404, 0.02406628243625164,
            -0.012850321829319,
            -0.01062958687543869, 0.0003391923673916608, 0.005292960442602634, -0.008390137925744057,
            -0.020161282271146774, 0.007229865528643131, -0.009070081636309624, 0.0015096009010449052,
            -0.022057639434933662, -0.0021209269762039185, -0.026199685409665108, 0.00018801783153321594,
            -0.024565324187278748, -0.008776894770562649, -0.023629620671272278, 0.01681770384311676,
            -0.014185258187353611, -0.02360466867685318, 0.0039549050852656364, -0.020136330276727676,
            -0.0013014069991186261, 0.0009294650517404079, -0.008627181872725487, 0.006992821116000414,
            0.002498327288776636, -0.016555706039071083, -0.010118069127202034, -0.004316709935665131,
            -0.02150869369506836, -0.0062380204908549786, 0.013237079605460167, -0.003917476627975702,
            0.02989259362220764, 0.23534803092479706, -0.00578576372936368, 0.0031018557492643595,
            0.03608071058988571,
            -0.003889405634254217, 0.024003902450203896, 0.024028854444622993, -0.00732967397198081,
            -0.013174699619412422, 0.0251267459243536, -0.0012608598917722702, 0.008015856146812439,
            -0.01648085005581379, 0.006774490233510733, 0.014110402204096317, 0.011384387500584126,
            -0.008483707904815674, 0.0015002439031377435, -0.01524572167545557, -0.03278703615069389,
            -0.012656943872570992, -0.006762014236301184, -0.016206376254558563, -0.004366614390164614,
            0.008945321664214134, 0.010560968890786171, -0.005174438003450632, 0.0035587907768785954,
            0.028470326215028763, 0.019188150763511658, -0.013711169362068176, -0.0030940582510083914,
            -0.006868060678243637, 0.022868582978844643, -0.013885833323001862, 0.004241853952407837,
            0.01789064332842827, 0.01512096170336008, 0.0029303100891411304, 0.0019525004317983985,
            0.0157572403550148,
            0.0008974951924756169, -0.014160306192934513, -0.017965499311685562, 0.012176616117358208,
            0.02159602753818035, 0.012756751850247383, 0.001325579360127449, -0.009631503373384476,
            0.03096553310751915,
            -0.032662276178598404, 0.0008951559429988265, 0.0111036766320467, 0.02477741800248623,
            -0.0019493814324960113, 0.011764906346797943, 0.0262994933873415, 0.007510576397180557,
            0.013049938715994358,
            0.00205854675732553, 0.010648300871253014, 0.03423425555229187, -0.027721762657165527,
            0.006500017363578081,
            -0.009095033630728722, 0.014784108847379684, -0.015482766553759575, 0.0015680823707953095,
            -0.0045568738132715225, 0.006387732923030853, -0.01659313403069973, 0.0005310114938765764,
            -0.0026495992206037045, 0.004085903521627188, -0.002454661065712571, -0.030167067423462868,
            0.015470290556550026, 0.022619063034653664, 0.008608467876911163, 0.02420351840555668,
            -0.026698727160692215,
            0.01916319876909256, -0.007173723541200161, -0.013536504469811916, -0.007005297113209963,
            -0.017453981563448906, 0.009656455367803574, -0.0029505835846066475, -0.024677608162164688,
            -0.001943143317475915, 0.0004082004597876221, -0.02921888791024685, -0.00046434265095740557,
            -0.011995713226497173, -0.0014877679059281945, 0.002431268570944667, 0.008907892741262913,
            0.03026687540113926, -0.0002970857312902808, -0.005040320567786694, -0.03168914467096329,
            -0.04291757941246033, 0.005383411422371864, 0.004566230811178684, 0.009668931365013123,
            -0.0181027352809906,
            0.004747133702039719, -0.007859906181693077, 0.004198187962174416, -0.010018261149525642,
            -0.010242830030620098, -0.0037303362041711807, 0.005476981867104769, 0.0029973688069730997,
            0.005520648322999477, -0.006116379052400589, 0.0013988760765641928, -0.016967415809631348,
            -0.011546575464308262, -0.00674330024048686, 0.003652360988780856, -0.020647848024964333,
            -0.020373374223709106, 0.010199163109064102, 0.0016234448412433267, -0.026598919183015823,
            0.00940693449229002, 0.011328245513141155, -0.009082557633519173, -0.04239358752965927,
            0.014172782190144062,
            -0.02079755999147892, 0.018539397045969963, 0.0005321811186149716, -0.004541278816759586,
            -0.0021848666947335005, -0.007616622839123011, -0.008458755910396576, 0.007323435973376036,
            0.018514445051550865, 0.014135354198515415, 0.005202508997172117, -0.010723156854510307,
            0.017553789541125298, 0.010560968890786171, -0.036155566573143005, 0.016505802050232887,
            -0.0054738628678023815, -0.02441561222076416, -0.013573932461440563, -0.017728453502058983,
            0.004831346683204174, -0.0043759713880717754, -0.011253388598561287, 0.03188876062631607,
            -0.03248761221766472, -0.011097438633441925, -0.02966802567243576, 0.004388447385281324,
            0.010997629724442959, -0.02886955812573433, -0.009562885388731956, 0.001766919274814427,
            -0.009675169363617897, -0.01559505145996809, -0.013561456464231014, -0.15829600393772125,
            -0.0035431955475360155, 0.015507718548178673, 0.006353423930704594, -0.008964035660028458,
            -0.010367590002715588, 0.040472276508808136, -0.020161282271146774, -0.028120996430516243,
            -0.013224603608250618, 0.012120474129915237, -0.0013653467176482081, -0.04027266055345535,
            -0.0016172068426385522, -0.023779334500432014, 0.005645408295094967, -0.023330196738243103,
            0.010080641135573387, 0.014646871946752071, 0.000837454223074019, 0.0078786201775074,
            -0.00865213479846716,
            0.018140163272619247, -0.0045038508251309395, -0.004191949963569641, 0.023879142478108406,
            -0.00922603253275156, 0.00805952213704586, -0.012257710099220276, -0.014185258187353611,
            -0.02767185866832733, -0.019662240520119667, -0.00892660766839981, -0.011247150599956512,
            -0.006687157787382603, 0.006375256925821304, -0.0056890747509896755, 0.001538451761007309,
            -0.020585468038916588, 0.031714096665382385, 0.022244781255722046, 0.04601163789629936,
            0.022257257252931595,
            0.01115981861948967, -0.0027587644290179014, 0.007036487106233835, 0.002510803285986185,
            0.02089736796915531,
            0.0056142183020710945, -0.0070926290936768055, -0.01680522784590721, -0.010317686013877392,
            0.022718871012330055, -0.004962345119565725, 0.004104617517441511, 0.005567433312535286,
            -0.004856299143284559, 0.01821501925587654, -0.012656943872570992, 0.014447255060076714,
            -0.016181424260139465, -0.01952500455081463, 0.002944345586001873, 0.0021178079769015312,
            -0.030192019417881966, -0.01670541800558567, -0.02361714467406273, 0.02522655390202999,
            -0.03166419267654419,
            -0.012762989848852158, -0.0006588909309357405, -0.005770168732851744, 0.013212127611041069,
            0.009619027376174927, -0.009993309155106544, 0.01530810259282589, -0.022731347009539604,
            0.004397804383188486, 0.003802073420956731, 0.01027402002364397, -0.004310471937060356,
            0.01243861299008131,
            -0.012214044108986855, -0.005651646293699741, 0.004921798128634691, -0.0029459050856530666,
            0.0005816954071633518, -0.003967381082475185, -0.003179830964654684, 0.0023657693527638912,
            -0.003986095078289509, -0.04304233938455582, -0.008496183902025223, -0.019999094307422638,
            0.0036554799880832434, 0.01554514653980732, 0.017666073516011238, 0.0004284740425646305,
            0.004166997503489256, 0.007067677099257708, -0.021945355460047722, -0.008751942776143551,
            -0.024739988148212433, 0.009662693366408348, 0.012001951225101948, 0.01512096170336008,
            0.024141138419508934,
            0.015707336366176605, 0.02231963723897934, -0.004048475064337254, -0.00532103143632412,
            0.024789893999695778,
            0.009893500246107578, 0.02056051604449749, -0.002091296250000596, -0.009088795632123947,
            0.006468827370554209, -0.004918679129332304, -9.771274199010804e-05, -0.008103188127279282,
            0.07066429406404495, 0.0005095683154650033, -0.026499111205339432, -0.004519445821642876,
            -0.009625265374779701, 0.022506777197122574, -0.10419989377260208, -0.02979278564453125,
            0.00420442596077919,
            0.007853668183088303, -0.006774490233510733, 0.015732288360595703, -0.007591670844703913,
            0.0035026485566049814, 0.006805680226534605, 0.021558599546551704, 0.010242830030620098,
            -0.0003674583858810365, -0.00028227042639628053, -0.026199685409665108, 0.03236284852027893,
            0.006038403604179621, 0.030416587367653847, -0.006418922916054726, -0.02407875843346119,
            0.018289875239133835, -0.006300400476902723, 0.013049938715994358, -0.0012249912833794951,
            -0.023292768746614456, -0.02102212980389595, -0.004138926509767771, -0.036280326545238495,
            0.00835894700139761, -0.0014752917923033237, -0.0030457135289907455, 0.008221711032092571,
            -0.010897821746766567, 0.005583028309047222, -0.0027946331538259983, 0.008976511657238007,
            0.017079699784517288, 3.0483452064800076e-05, -0.011833524331450462, 0.018526921048760414,
            -0.015008676797151566, -0.0012881512520834804, -0.0026823487132787704, 0.010118069127202034,
            -0.023317720741033554, 0.023430004715919495, -0.004282400943338871, 0.016069140285253525,
            0.01845206506550312, -0.001513499766588211, -0.007111343089491129, -0.028071092441678047,
            -0.005913643166422844, -0.021645931527018547, -0.015582575462758541, 0.04593678191304207,
            0.009144937619566917, 0.014334971085190773, 0.02209506928920746, 0.00713005755096674,
            0.02032347023487091,
            0.016917511820793152, -0.011764906346797943, -0.009057605639100075, 0.03835134953260422,
            0.033111412078142166, -0.0037864784244447947, -0.021471265703439713, 0.005820073187351227,
            0.00807199813425541, -0.005330388434231281, -0.029019271954894066, 0.038725629448890686,
            -0.01028649602085352, 0.023991426452994347, -0.019350338727235794, 0.00038773196865804493,
            -0.015931904315948486, -0.03909991309046745, 0.01858930103480816, -0.007504338398575783,
            -0.020959747955203056, -0.023554764688014984, -0.0038987628649920225, -0.021209269762039185,
            0.025189125910401344, 0.016256282106041908, -0.014596967957913876, 0.02138393372297287,
            -0.003452744334936142, 0.006231782492250204, 0.008190521039068699, 0.012413660995662212,
            0.010386303998529911, -0.04653563350439072, -0.027272624894976616, 0.010904059745371342,
            0.004425875376909971, -0.005891810171306133, 0.002186426194384694, 0.015807144343852997,
            -0.018763964995741844, -0.01332441158592701, -0.04097132012248039, 0.016630562022328377,
            -0.013960689306259155, -0.013511552475392818, 0.018414637073874474, 0.011540337465703487,
            -0.014097926206886768, -0.018140163272619247, -0.011902143247425556, -0.0111036766320467,
            -0.006228663492947817, 0.003209461458027363, -0.014447255060076714, 0.008683324791491032,
            0.01634361408650875, -0.019787000492215157, 0.0035151245538145304, -0.01319965161383152,
            0.004460184834897518, 0.007591670844703913, 0.009606551378965378, -0.007785049732774496,
            0.021408885717391968, 0.0007193217170424759, -0.002899120096117258, -0.00017641900922171772,
            -0.002007083036005497, 0.012295138090848923, -0.028595086187124252, -0.011234674602746964,
            -0.01449715904891491, -0.04174483194947243, 0.006836870219558477, 0.025650739669799805,
            0.0014916666550561786, -0.013711169362068176, 0.004120212513953447, -0.00564228929579258,
            0.003792716423049569, 0.011265864595770836, -0.010978915728628635, -0.022856106981635094,
            0.00713005755096674, -0.039074961096048355, -0.03256246820092201, -0.004821989685297012,
            0.00023801946372259408, 0.0024702560622245073, 0.020510612055659294, 0.004142045509070158,
            0.017703501507639885, 0.020834987983107567, -0.01683017984032631, -0.007398292422294617,
            0.001146236201748252, -0.028120996430516243, 0.004909322131425142, -0.015919428318738937,
            -0.004098379518836737, -0.038625821471214294, 0.015071057714521885, -0.002375126350671053,
            0.013773549348115921, 0.026948248967528343, 0.015956856310367584, -0.0007544105756096542,
            0.0023501741234213114, -0.021758215501904488, 0.008471231907606125, -0.023529812693595886,
            -0.01286279782652855, 0.025975117459893227, 0.0034496253356337547, 0.008115665055811405,
            0.006624777801334858, -0.0023252221290022135, -0.0035151245538145304, 0.007173723541200161,
            -0.017678549513220787, 0.007834953255951405, 0.02919393591582775, -0.00709886709228158,
            -0.0049249171279370785, 0.017553789541125298, 0.020597944036126137, 0.00532103143632412,
            -0.02207011543214321, -0.022456873208284378, -0.002250365912914276, 0.023392576724290848,
            -0.0169299878180027, -0.0022534849122166634, -0.005361578427255154, 0.0021240459755063057,
            -0.013848405331373215, 0.00479079969227314, -0.02288105897605419, 0.006974106654524803,
            -0.005679717753082514, 0.012114236131310463, 0.0038582156412303448, -0.0016250043408945203,
            -0.01115981861948967, -0.02779661864042282, -0.0008928166935220361, 0.006992821116000414,
            -0.009600313380360603, -0.023554764688014984, -0.0014316256856545806, 0.006228663492947817,
            0.0006238020723685622, -0.02185802347958088, -0.011827286332845688, 0.015283149667084217,
            -0.010710680857300758, 0.033360932022333145, 0.0025341957807540894, -0.02852023020386696,
            -0.02906917594373226, 0.034883011132478714, 0.014784108847379684, 0.001349751721136272,
            0.024465516209602356,
            -0.00888917874544859, 0.010654538869857788, 0.005146367009729147, 0.01800292730331421,
            -0.033485691994428635,
            -0.029942497611045837, 0.009244746528565884, -0.013648788444697857, -0.002880405867472291,
            -0.0024094353429973125, 0.00812190305441618, -0.008514897897839546, -0.014222686178982258,
            0.010673252865672112, 0.03588109463453293, -0.009787454269826412, 0.05249917879700661,
            0.026424255222082138,
            -0.01717950776219368, -0.004597421269863844, 0.007429482415318489, 0.0049249171279370785,
            0.013848405331373215, 0.02476494200527668, -0.036280326545238495, -0.013000034727156162,
            0.004519445821642876, -0.004166997503489256, 0.010542254894971848, -0.02557588368654251,
            -0.015008676797151566, -0.022469349205493927, -0.03343578800559044, -0.014434779062867165,
            -0.009213556535542011, -0.015445338562130928, 0.019013486802577972, 0.0003068326332140714,
            0.015033629722893238, -0.0028071091510355473, -0.02430332824587822, -0.021446313709020615,
            0.026549015194177628, 0.002520160283893347, -0.0023610906209796667, -0.021645931527018547,
            -0.002995809307321906, 0.0016733489464968443, -0.02244439721107483, -0.007647812832146883,
            0.027172816917300224, -2.9581862690974958e-05, 0.01880139298737049, 0.013923261314630508,
            0.02487722598016262, -0.0031782714650034904, -0.00899522565305233, 0.010897821746766567,
            -0.015158389694988728, -0.008920369669795036, 0.018177591264247894, 0.04366614297032356,
            -0.027896426618099213, 0.021571075543761253, -0.03633023053407669]
    # "A sharp knife in a bag"
    arr2 = [0.010112764313817024, -0.001511418609879911, 0.01431563962250948, 0.003040618496015668,
            -0.008735514245927334,
            0.01339747291058302, -0.020497098565101624, -0.006035975180566311, 0.0004962625680491328,
            -0.016333019360899925, 0.030726250261068344, 0.025230182334780693, -0.0030761812813580036,
            0.004047692287713289, -0.006711668334901333, 0.019759979099035263, 0.03690771013498306,
            0.014250979758799076,
            0.031321119517087936, -0.020484168082475662, -0.019035791978240013, 0.030959025025367737,
            0.016501134261488914,
            -0.006352807395160198, -0.0031537727918475866, 0.006446564104408026, 0.008360488340258598,
            -0.021829087287187576, -0.0023309793323278427, -0.004089721012860537, 0.009673078544437885,
            -0.008657922968268394, -0.028941646218299866, -0.017626212909817696, -0.013462132774293423,
            0.012556897476315498, -0.0044518145732581615, -0.02116956003010273, 0.005311787594109774,
            -0.013500927947461605, 0.03781294450163841, 0.01624249666929245, 0.0037955197039991617,
            -0.0015599132748320699,
            -0.0011355845490470529, 0.00866438914090395, -0.012802604585886002, 0.010959804989397526,
            -0.03923545777797699,
            0.024906884878873825, 0.024531859904527664, -0.01348799653351307, -0.029303738847374916,
            0.007319469004869461,
            0.015246737748384476, 0.012078416533768177, 0.01862196996808052, -0.0039345379918813705,
            0.014367367140948772,
            -0.015531240031123161, 0.002031928626820445, 0.007416458334773779, -0.0187771525233984,
            -0.006278448738157749,
            0.010132161900401115, -0.003481920575723052, -0.011632265523076057, 0.015401921235024929,
            -0.015039827674627304, 0.016720976680517197, 0.018001237884163857, 0.017471028491854668,
            0.01603558473289013,
            -0.008341090753674507, 0.006427166052162647, 0.004603764973580837, -0.022527411580085754,
            0.022411024197936058,
            0.017652075737714767, 0.0012414646334946156, 0.010655905120074749, -0.026769082993268967,
            -0.024531859904527664, 0.008056588470935822, 0.027881227433681488, -0.03486446663737297,
            -0.03305399790406227,
            0.01604851707816124, -0.020587623119354248, -0.0009213995654135942, 0.012931923381984234,
            0.0230058915913105,
            -0.002694689668715, 0.021117830649018288, -0.021984269842505455, 0.02388526313006878,
            -0.006462729070335627,
            0.0295882411301136, -0.0057644047774374485, -0.019397886469960213, -0.0014920206740498543,
            0.007894939742982388, -0.022333431988954544, -0.009711874648928642, -0.012007291428744793,
            -0.023277463391423225, -0.013669043779373169, -0.030312430113554, 0.013500927947461605,
            -0.003717927960678935,
            -0.0409424714744091, 0.014108728617429733, 0.012304725125432014, -0.04120110720396042,
            0.015919197350740433,
            -0.0022420722525566816, 0.013087106868624687, -0.017108935862779617, -0.004358058329671621,
            -0.03101075254380703, 0.011250773444771767, 0.01917804218828678, 0.023419713601469994,
            -0.02061348594725132,
            0.019850503653287888, 0.012944855727255344, -0.028398504480719566, -0.009356247261166573,
            -0.009078210219740868, -0.014548414386808872, 0.044796183705329895, 0.024777565151453018,
            0.016902023926377296,
            0.00570297846570611, -0.01222066767513752, 0.03137284517288208, -0.026898400858044624,
            0.004813908599317074,
            -0.008192373439669609, -0.012977185659110546, -0.0042158071883022785, 0.016552861779928207,
            -0.033674728125333786, -0.009789465926587582, -0.01449668686836958, 0.0031359915155917406,
            0.016345951706171036, 0.026161281391978264, 0.009563157334923744, -0.026950128376483917,
            0.024764634668827057,
            -0.01449668686836958, 0.01841505989432335, -0.01731584593653679, -0.0007233794895000756,
            0.0066405427642166615,
            0.009621351025998592, -0.014031137339770794, -0.02539829909801483, -0.009918785654008389,
            -0.004044459201395512, 0.006094168871641159, 0.029122691601514816, -0.008826037868857384,
            -0.010752894915640354, 0.015169146470725536, 0.0021693301387131214, -0.014393230900168419,
            0.0020594089291989803, 0.014807052910327911, 0.010642972774803638, 0.009155802428722382,
            -0.01804003305733204,
            0.021997202187776566, 0.005085479002445936, -0.0036823651753365993, 0.01897113211452961,
            0.006145896855741739,
            -0.02663976326584816, -0.008606195449829102, -0.01659165881574154, -0.0024085708428174257,
            0.008451011963188648, 0.0317608043551445, -0.005596289876848459, -0.016164904460310936,
            0.020199663937091827,
            0.013184096664190292, 0.0001768036454450339, 0.005234196316450834, 0.009453236125409603,
            0.02208772487938404,
            -0.005098410882055759, -0.02135060541331768, -0.6914958357810974, -0.03044174797832966,
            -0.0026461950037628412,
            -0.019100451841950417, 0.03703702986240387, 0.02502327226102352, 0.02551468461751938,
            0.01341040525585413,
            0.019268566742539406, 0.015945062041282654, -0.00669873645529151, 0.025993166491389275,
            0.006029509473592043,
            -0.01643647439777851, -0.011199045926332474, -0.012640955857932568, 0.01048779021948576,
            -0.020109141245484352,
            -0.013423336669802666, 0.011839176528155804, -0.02519138716161251, 0.029795153066515923,
            -0.017587415874004364,
            0.002201660070568323, 0.021389402449131012, 0.010313209146261215, 0.02484222501516342,
            -0.02296709641814232,
            0.013177630491554737, 0.00840575061738491, -0.015298466198146343, 0.006029509473592043,
            -0.005357049405574799,
            0.002159631345421076, 0.047873981297016144, -0.017962442710995674, -0.005030518397688866,
            0.030933162197470665,
            0.02229463681578636, 0.01569935493171215, -0.011050328612327576, -0.002996973227709532,
            0.004228739067912102,
            -0.01496223546564579, -0.042416710406541824, -0.01898406445980072, 0.028605414554476738,
            -0.0025653704069554806, 0.014457890763878822, -0.011832710355520248, 0.025062067434191704,
            0.010061036795377731, 0.0005605180631391704, -0.014522550627589226, 0.010849883779883385,
            0.011386558413505554,
            0.02168683521449566, -0.02047123573720455, 0.020199663937091827, -0.009543759748339653,
            -0.017083071172237396,
            0.02097558043897152, -0.00802425853908062, -0.015363125130534172, -0.013953546062111855,
            -0.004109118599444628,
            -0.019475476816296577, -0.0019155412446707487, 0.01104386243969202, -0.014703596942126751,
            0.01935908943414688,
            0.005864627193659544, -0.0011363928206264973, -0.00529885571449995, 0.007481117732822895,
            0.03483860194683075,
            -0.0033057229593396187, -0.0012050936929881573, -0.00515337148681283, -0.011748652905225754,
            0.012110746465623379, 0.0014637321000918746, -0.021945474669337273, 0.0042966315522789955,
            0.028320912271738052, -0.0012834934750571847, 0.0067957257851958275, 0.002371391514316201,
            0.01514328271150589,
            -0.0026332628913223743, -0.006782793905586004, 0.02442840300500393, -0.009673078544437885,
            -0.012834934517741203, -0.0018088528886437416, 0.014755325391888618, -0.0077074263244867325,
            0.010468392632901669, 0.024143900722265244, 0.0034786874894052744, -0.014819984324276447,
            0.0233679860830307,
            0.017419300973415375, 0.006126498803496361, 0.0025233416818082333, 0.015427784994244576,
            0.017289981245994568,
            0.008198839612305164, 0.02277311682701111, -0.02133767493069172, -0.010733496397733688,
            -0.0017781394999474287,
            0.004199642222374678, -0.023264531046152115, 0.0056092217564582825, -0.02609662152826786,
            0.006970306858420372,
            0.018828880041837692, 0.022113589569926262, -0.026019031181931496, 0.0050078872591257095,
            -0.011768050491809845, 0.036959439516067505, -0.006743998266756535, 0.012809070758521557,
            0.025268979370594025,
            -0.019475476816296577, -0.0032475292682647705, -0.026174213737249374, -0.01404406875371933,
            -0.0014613074017688632, 0.010345539078116417, 0.003176403697580099, -0.010190355591475964,
            0.002138616982847452, -0.0032927910797297955, -0.010714098811149597, -0.013604383915662766,
            0.00598101457580924, -0.02005741372704506, -0.01642354391515255, -0.006242886185646057,
            -0.009563157334923744,
            -0.019307361915707588, -0.0023342121858149767, -0.010649438947439194, 0.007772086188197136,
            -0.008522137999534607, -0.013565587811172009, -0.01440616324543953, 0.013067709282040596,
            -0.004613463766872883, -0.01894526742398739, -0.021130762994289398, 0.01881594955921173,
            -0.01148354820907116,
            -0.024389607831835747, -0.0007027692045085132, -0.010461926460266113, -0.014470823109149933,
            0.025450026616454124, 0.03261431306600571, -0.02136353775858879, 0.004467979539185762,
            -0.03333849832415581,
            -0.01863490231335163, 0.043890949338674545, 0.017225323244929314, -0.005363515578210354,
            -0.043916814029216766,
            0.01348799653351307, -0.017665008082985878, -0.0004962625680491328, 0.00986059196293354,
            -0.0065888152457773685, 0.007164285518229008, -0.024893952533602715, 0.026187146082520485,
            0.025631071999669075, 0.005159837659448385, -8.572451042709872e-05, 0.0005985056050121784,
            -0.019656524062156677, -0.0022614700719714165, 0.012388782575726509, 0.021958407014608383,
            -0.009362712502479553, 0.03160561993718147, 0.012162473984062672, 0.006223488133400679,
            0.005340884439647198,
            -0.007668630685657263, 0.000798142165876925, -0.009117006324231625, -0.01677270419895649,
            -0.018583174794912338, 0.021609244868159294, 0.016526998952031136, 0.01513035036623478,
            0.01588040217757225,
            0.027131175622344017, -0.023109348490834236, -0.000978784984908998, 0.0018573475535959005,
            0.023096416145563126, -0.03817503899335861, 0.020341916009783745, -0.020199663937091827,
            0.033855777233839035,
            0.03344195336103439, -0.006546786520630121, -0.02133767493069172, 0.012078416533768177,
            -0.0020594089291989803,
            -0.01130250096321106, 0.02772604487836361, -0.007422924041748047, 0.015350193716585636,
            0.004328961484134197,
            0.0037534907460212708, 0.0028628045693039894, -0.010662371292710304, 0.013669043779373169,
            -0.0037761216517537832, -0.016358884051442146, 0.008522137999534607, 0.00783674605190754,
            0.018531447276473045,
            -0.016488201916217804, -0.038226768374443054, -0.021790292114019394, 0.021389402449131012,
            0.01422511599957943,
            0.015621763654053211, 0.006168527528643608, -0.00029682807507924736, 0.013462132774293423,
            -0.01440616324543953, 0.013332813046872616, 0.024557722732424736, -0.00792726967483759,
            -0.005350583232939243,
            0.025773324072360992, -0.005706211552023888, 0.019979821518063545, 0.008690252900123596,
            0.02992447093129158,
            0.007377662695944309, 0.0011024464620277286, -0.010998601093888283, 0.009175200015306473,
            -0.013054776936769485, -0.0021046705078333616, 0.006524155382066965, 0.011095590889453888,
            -0.036054205149412155, 0.024389607831835747, 0.021596312522888184, -0.0006401302525773644,
            0.02097558043897152,
            -0.009084676392376423, 0.0037567238323390484, -0.004474445711821318, -0.01880301721394062,
            0.01131543330848217,
            -0.016113176941871643, -0.004054157994687557, -0.015169146470725536, -0.024919817224144936,
            0.01569935493171215, 0.018570242449641228, -0.018376262858510017, -0.0005839571822434664,
            0.004070322960615158,
            -0.0005302088684402406, -3.748237213585526e-05, 0.01531139761209488, 0.021660972386598587,
            -0.011690459214150906, 0.000496666703838855, -0.025230182334780693, -0.027234630659222603,
            0.0119555639103055,
            0.005622153636068106, -0.017057206481695175, -0.012996583245694637, -0.007293604779988527,
            0.016345951706171036, 0.002138616982847452, 0.03044174797832966, 0.013203494250774384,
            0.02992447093129158,
            -0.008004860952496529, 0.0119555639103055, 0.0030826472211629152, -0.00724834343418479,
            0.010371402837336063,
            -0.017289981245994568, -0.004170545376837254, 0.010436062701046467, -0.0039410036988556385,
            -0.03354540839791298, -0.022372227162122726, -0.0301313828676939, 0.02718290314078331,
            0.013203494250774384,
            -0.0075069814920425415, -0.0063140117563307285, 0.01522087398916483, 0.013604383915662766,
            0.013927682302892208, 0.001208326662890613, -0.0020707242656499147, -0.0035239493008702993,
            0.012459908612072468, 0.0009513046243228018, -0.020135005936026573, 0.025605209171772003,
            0.019953958690166473,
            0.02041950821876526, -0.015828674659132957, -0.020497098565101624, -0.024674110114574432,
            0.015466581098735332,
            0.08167803287506104, 0.02751913294196129, -0.003456056583672762, 0.021764427423477173,
            0.008347556926310062,
            -0.021402332931756973, -0.016514066606760025, 0.014548414386808872, 0.007228945381939411,
            0.0054023112170398235, 0.016320087015628815, 0.00711902417242527, -0.009976979345083237,
            -0.009013551287353039,
            0.003378465073183179, -0.015945062041282654, -0.00877431035041809, -0.006776328198611736,
            0.00792726967483759,
            -0.03044174797832966, -0.0018589639803394675, -0.012168940156698227, -0.0038601793348789215,
            -0.0013263304717838764, -0.015298466198146343, -0.008890697732567787, 0.013384541496634483,
            0.03434718772768974, 0.028527824208140373, -0.01166459545493126, 0.014755325391888618,
            0.0027221699710935354,
            0.002500710776075721, 0.0050078872591257095, -0.017794327810406685, -0.014923440292477608,
            0.011289569549262524, -0.008632059209048748, 0.012078416533768177, -0.002161247655749321,
            0.002266319701448083,
            -0.006074771285057068, 0.016643386334180832, 0.00332997040823102, 0.010196821764111519,
            0.0041414485312998295,
            -0.010326141491532326, 0.013772498816251755, -0.007377662695944309, -0.028553687036037445,
            0.030053790658712387, 0.021570449694991112, -0.009162267670035362, 0.010313209146261215,
            0.0321228988468647,
            0.03341609239578247, 0.011587003245949745, 0.018841812387108803, 0.004025061149150133,
            -0.019281499087810516,
            0.0002390385343460366, -0.008004860952496529, 0.015582968480885029, -0.009149336256086826,
            -0.004212574101984501, -0.02369128353893757, -0.017238253727555275, 0.005049915984272957,
            -0.00865145679563284,
            -0.008670855313539505, 0.0008429997833445668, -0.015518308617174625, -0.016850296407938004,
            -0.005841996520757675, 0.0021030541975051165, 0.0019979821518063545, 0.018557310104370117,
            -0.00921399611979723, 0.0003447165945544839, 0.006495058536529541, 0.004794510547071695,
            -0.021027307957410812,
            0.014380299486219883, -0.04029587283730507, -0.004047692287713289, 0.002528191078454256,
            0.006873317528516054,
            -0.02353610098361969, -0.0193202942609787, 0.012298259884119034, 0.024712907150387764,
            0.012156008742749691,
            0.010520120151340961, -0.0027641986962407827, -0.0217385645955801, -0.010339072905480862,
            0.013203494250774384,
            0.02442840300500393, 0.0035983077250421047, -0.005040217190980911, 0.014199252240359783,
            -0.01440616324543953,
            -0.010662371292710304, -0.015479512512683868, 0.004788044840097427, 0.000659123994410038,
            0.004720152355730534,
            -0.0034948524553328753, -0.0056771147064864635, 0.03406268730759621, 0.01077875867486,
            -0.01954013668000698,
            0.006724600214511156, -0.0029048332944512367, 0.008237635716795921, 0.025644004344940186,
            0.005845229607075453,
            0.007810881827026606, 0.0069897049106657505, -0.003931304905563593, -0.005709444172680378,
            -0.0028886685613542795, 0.0016148739960044622, 0.0035950748715549707, -0.00830229464918375,
            0.006136197596788406, 0.005399078130722046, -0.03064865991473198, -0.020147936418652534,
            0.009285121224820614,
            -0.014018204994499683, -0.003924838732928038, -0.01716066338121891, 0.015272601507604122,
            0.003420493798330426,
            -0.03721807524561882, -0.0005892107728868723, 0.027053585276007652, -0.030002063140273094,
            -0.026161281391978264, -0.030027925968170166, -0.018169352784752846, -0.018130557611584663,
            0.011114988476037979, -0.030415885150432587, -0.017483960837125778, 0.00018387578893452883,
            -0.002405337756499648, -0.014832916669547558, 0.009511429816484451, -0.0025799188297241926,
            -0.0009779767133295536, -0.018117625266313553, -0.008806640282273293, -0.010642972774803638,
            -0.03416614234447479, -0.02661389857530594, 0.010979203507304192, 0.016100244596600533,
            0.04073556140065193,
            0.05014999955892563, -0.01606144942343235, 0.014419094659388065, 0.013345745392143726,
            -0.00822470337152481,
            -0.003420493798330426, -0.015414852648973465, -0.010752894915640354, -0.01841505989432335,
            0.021130762994289398, 0.024751702323555946, -0.02208772487938404, -0.0011671060929074883,
            0.010888679884374142,
            -0.01717359386384487, 0.021790292114019394, 0.007739756256341934, -0.01084341760724783,
            -3.298650699434802e-05,
            -0.017005478963255882, -0.021751495078206062, -0.025592276826500893, -0.0021790291648358107,
            0.011994359083473682, -0.02718290314078331, -0.016902023926377296, 0.014664801768958569,
            0.012071950361132622,
            0.021842019632458687, 0.009485566057264805, 0.01769087091088295, 0.0035692108795046806,
            0.008476875722408295,
            0.0018977598519995809, -0.006491825915873051, -0.0012341904221102595, -0.01278320699930191,
            0.02065228298306465, 0.015026895329356194, 0.00743585592135787, 0.02228170447051525,
            -0.010727031156420708,
            -0.0129513218998909, -0.03447650745511055, 0.0020125305745750666, 0.03318331763148308,
            -0.005754705984145403,
            0.008354023098945618, 0.012375851161777973, -0.00665994081646204, -0.031114207580685616,
            -0.0299503356218338,
            -0.02775190770626068, -0.03628697618842125, 0.018195217475295067, 0.004671657457947731,
            0.003915139939635992,
            0.03240739926695824, -0.02224290929734707, 0.0020577923860400915, -0.0007270165951922536,
            -0.005169536452740431, 0.028217457234859467, -0.014354435727000237, 0.023109348490834236,
            0.01194263156503439,
            -0.007468185853213072, -0.0075069814920425415, 0.02479049749672413, 0.002327746246010065,
            -0.02682081051170826,
            -0.007720358669757843, 0.012265929952263832, 0.010888679884374142, -0.005706211552023888,
            0.004700754303485155,
            0.020160868763923645, -0.013436269015073776, -0.006718134507536888, 0.02356196567416191,
            0.005369981285184622,
            0.018117625266313553, -0.031269390136003494, -0.014031137339770794, -0.02247568406164646,
            0.017419300973415375,
            -0.005664182361215353, 0.00761690316721797, -0.00198666681535542, -0.0033558341674506664,
            -0.013992341235280037, 0.012964253313839436, -0.017781395465135574, 0.04065796732902527,
            0.011535275727510452,
            -0.019022859632968903, -0.004739549942314625, -0.013979409821331501, 0.001466156798414886,
            0.01004163920879364,
            -0.003911906853318214, 0.015789879485964775, -0.0011331598507240415, 0.014108728617429733,
            -0.010106298141181469, 0.0013133984757587314, -0.0057579390704631805, 0.005295622628182173,
            -0.0006684188265353441, 0.02229463681578636, -0.005874326452612877, -0.016087312251329422,
            -0.008366954512894154, -0.01679856888949871, 0.005670648533850908, 0.012815535999834538,
            -0.005951917730271816,
            -0.012097815051674843, 0.017613280564546585, -0.01970825158059597, 0.009078210219740868,
            0.008522137999534607,
            -0.002216208493337035, 0.0015518307918682694, -0.017238253727555275, 0.004464746452867985,
            -0.027415677905082703, -0.01422511599957943, 0.010849883779883385, -0.014768256805837154,
            -0.036054205149412155, -0.002993740374222398, -0.006187925580888987, -0.009647214785218239,
            0.006065072026103735, -0.008425148203969002, -0.0031165936961770058, 0.028605414554476738,
            -0.022656729444861412, 0.0028902848716825247, 0.0026090156752616167, 0.03848540410399437,
            -0.015195010229945183, -0.004894732963293791, 0.004797743633389473, 0.0029403960797935724,
            0.031476303935050964, -0.02754499763250351, -0.009905853308737278, -0.011451218277215958,
            -0.01754862070083618,
            0.028424369171261787, 0.007901404984295368, -0.012776740826666355, 0.01679856888949871,
            -0.005479902494698763,
            0.005670648533850908, -0.004047692287713289, -0.010862816125154495, 0.015815742313861847,
            -0.050434503704309464, 0.013694907538592815, 0.007461720146238804, -0.009627817198634148,
            -0.04451168328523636,
            -0.009744204580783844, 0.024234425276517868, -2.0932288862240966e-06, -0.01487171184271574,
            -0.015647627413272858, 0.025281911715865135, 0.03150216490030289, 0.0021450829226523638,
            -0.011671061627566814,
            0.02168683521449566, 0.020354848355054855, -0.020199663937091827, -0.005457271821796894,
            0.008683786727488041,
            -0.0018832114292308688, 0.013927682302892208, 0.016720976680517197, -0.021673904731869698,
            0.013235824182629585, -0.009065278805792332, 0.0006793301436118782, -0.006737532094120979,
            -0.018583174794912338, -0.023070551455020905, -0.006187925580888987, -0.028165729716420174,
            0.013811294920742512, -0.004885034170001745, -0.021298877894878387, -0.0050466833636164665,
            -0.021466992795467377, 0.001486363005824387, -0.0014402930391952395, -0.019785843789577484,
            0.009640749543905258, 0.013669043779373169, 0.024337880313396454, -0.00478481175377965,
            0.023406781256198883,
            -0.0004128112632315606, 0.018091760575771332, 0.01567349210381508, 0.004164079669862986,
            0.027131175622344017,
            0.005864627193659544, 0.017988305538892746, -0.00011143681331304833, -0.02043243870139122,
            -0.03582143038511276, -0.018492650240659714, 0.018479719758033752, 0.012026689015328884,
            0.015440717339515686,
            -0.029096828773617744, -0.020665213465690613, -0.009983445517718792, -0.0010151560418307781,
            0.007494049612432718, 0.018169352784752846, -0.002099821111187339, -0.023432645946741104,
            -0.00629461370408535,
            0.005748240277171135, 0.018764222040772438, -0.01827280782163143, -0.005036984104663134,
            0.026394056156277657,
            -0.016475271433591843, -0.002793295541778207, -0.01791071519255638, -0.00015811297635082155,
            0.011813312768936157, -0.012996583245694637, 0.03597661107778549, -0.008244100958108902,
            0.007222479209303856,
            0.008619126863777637, 0.017070138826966286, -0.013850090093910694, -0.019772911444306374,
            -0.00038977625081315637, -0.03362300246953964, -0.020665213465690613, 0.01569935493171215,
            -0.010235617868602276, 0.018583174794912338, -0.02684667333960533, 0.000624773558229208,
            -0.009504963643848896,
            -0.0013861405896022916, 0.010901611298322678, -0.020548826083540916, -0.007209547329694033,
            0.01589333452284336, -0.015518308617174625, 0.003481920575723052, 0.01767794042825699,
            0.02279898151755333,
            -0.021673904731869698, 0.026135418564081192, -0.015013962984085083, 0.010242084041237831,
            -0.0021709466818720102, 0.006866851355880499, -0.002389173023402691, -0.014807052910327911,
            -0.024130970239639282, 0.006074771285057068, -0.01260215975344181, 0.012944855727255344,
            0.028372639790177345,
            0.2238774597644806, 0.0008737131138332188, -0.010326141491532326, 0.039675142616033554,
            0.01306124310940504,
            0.012207736261188984, 0.0237300805747509, -0.004752481821924448, 0.009905853308737278,
            0.013565587811172009,
            0.011179648339748383, 0.004167312290519476, -0.01571228727698326, 0.0033170385286211967,
            0.003983032424002886,
            0.0025249579921364784, -0.027338087558746338, -0.03411441296339035, -0.006259051151573658,
            -0.024053378030657768, 0.011974961496889591, 0.007797949947416782, 0.011050328612327576,
            0.006520922761410475,
            0.019307361915707588, 0.006879783235490322, -0.017251186072826385, 0.012446976266801357,
            0.002083656145259738,
            -0.0036726663820445538, 0.0039410036988556385, -0.010009309276938438, 0.022721389308571815,
            -0.02464824728667736, 0.0024619149044156075, 0.0044227177277207375, -0.0019252401543781161,
            0.014354435727000237, 0.016268359497189522, 0.013643179088830948, 0.004083254840224981,
            0.006391603499650955,
            0.004458280745893717, -0.01567349210381508, -0.026743218302726746, 0.0018153188284486532,
            -0.01734171062707901,
            -0.007823813706636429, 0.011612867936491966, 0.007797949947416782, -0.005835530813783407,
            -0.010681768879294395, 0.006520922761410475, -0.008522137999534607, -0.0031844861805438995,
            -0.004047692287713289, -0.0030761812813580036, -0.005984247662127018, 0.0008195606642402709,
            -0.00382784940302372, 0.004914131015539169, 0.024363745003938675, -0.02793295495212078,
            0.012013757601380348,
            -0.011845641769468784, 0.0328470878303051, -0.03134698420763016, 0.003704996081069112,
            0.014677733182907104,
            -0.0195013415068388, 0.008826037868857384, -0.015001031570136547, -0.01791071519255638,
            0.01267975103110075,
            -0.018764222040772438, -0.03685598447918892, 0.0014411011943593621, 0.021247150376439095,
            0.012485772371292114,
            0.02570866420865059, -0.016384746879339218, -0.0077462224289774895, 0.00212406856007874,
            0.003970100544393063,
            -0.02022552862763405, -0.04099419713020325, -0.002709238091483712, -0.0041964091360569,
            -0.008263499476015568,
            -0.020665213465690613, 0.012479306198656559, -0.025993166491389275, -0.0003538093587849289,
            -0.0006405343301594257, 0.01912631466984749, 0.027079448103904724, 0.0012649038108065724,
            0.008172975853085518,
            0.003911906853318214, 9.15590305794467e-07, -0.019294429570436478, -0.021479925140738487,
            -0.002096588024869561, -0.018195217475295067, -0.0023503771517425776, -0.0004095782642252743,
            -0.030984889715909958, 0.020354848355054855, 0.027312222868204117, -0.026006098836660385,
            -0.011658129282295704, -0.03455410152673721, 0.006048907525837421, 0.003611239604651928,
            0.008632059209048748,
            0.015013962984085083, -0.00502081960439682, -0.02005741372704506, -0.02098851278424263,
            -0.02062641829252243,
            0.023652488365769386, -0.018583174794912338, 0.024312017485499382, 0.0025346570182591677,
            -0.006304312963038683, 0.00045948740444146097, -0.00019246339797973633, 0.0073517984710633755,
            -0.0014257446164265275, -0.04086487740278244, 0.011987892910838127, -0.019811706617474556,
            0.0047718798741698265, -0.006847453769296408, 0.002854722086340189, 0.0021483157761394978,
            0.006252584978938103, -0.00940150860697031, 0.0115934694185853, -0.0029193817172199488,
            -0.000670035311486572,
            0.0011687226360663772, -0.0002883414854295552, -0.016164904460310936, 0.0342954620718956,
            -0.008328159339725971, 0.019242702051997185, 0.0023034990299493074, 0.0005512232310138643,
            -0.014548414386808872, -0.018195217475295067, -0.009556692093610764, -0.0386405885219574,
            -0.03902854397892952,
            -0.010461926460266113, -0.01899699680507183, -0.03028656542301178, -0.02047123573720455,
            0.011515878140926361,
            -0.008425148203969002, -0.007953133434057236, -0.02904510125517845, 0.042959850281476974,
            -0.005350583232939243, 0.0097506707534194, -0.021958407014608383, -0.16635626554489136,
            0.01642354391515255,
            0.016087312251329422, -0.04448581859469414, 0.020186733454465866, 0.011399490758776665,
            -0.005508999340236187,
            0.007002636790275574, -0.0030923462472856045, -0.0015607215464115143, -0.0008551234495826066,
            0.013617315329611301, -0.019992753863334656, -0.022165317088365555, 0.015117418952286243,
            -0.016113176941871643, 0.004070322960615158, 0.03739912435412407, 0.0187771525233984,
            0.02480342984199524,
            0.015104486607015133, -0.008366954512894154, 0.008263499476015568, -0.01194263156503439,
            0.008741980418562889,
            -0.00611033383756876, 0.0014766640961170197, 0.023652488365769386, -0.007810881827026606,
            0.0009909087093546987, -0.0030891133937984705, -0.008522137999534607, 0.0010232384083792567,
            0.02133767493069172, 0.00010961826046695933, -0.021273015066981316, -0.007009102497249842,
            -0.019009927287697792, -0.011082658544182777, 0.02863127924501896, 0.0032345973886549473,
            0.03589902073144913,
            -0.010681768879294395, -0.0005269758985377848, 0.010545983910560608, 0.01677270419895649,
            0.006265516858547926,
            -0.010287345387041569, -0.02133767493069172, -0.024143900722265244, 0.006310778670012951,
            -0.000325924891512841, 0.006957374978810549, 0.008179442025721073, 0.021117830649018288,
            0.016087312251329422,
            0.017108935862779617, -0.015751082450151443, -0.008244100958108902, -0.006627610884606838,
            -0.006372205447405577, -0.011056794784963131, -0.007041432429105043, -0.004904432222247124,
            -0.03556279093027115, -0.016850296407938004, -0.0075522433035075665, -0.017626212909817696,
            0.005968082696199417, 0.002293800003826618, 0.003743791952729225, 0.016889091581106186,
            0.024738769978284836,
            -0.008819571696221828, -0.0002883414854295552, 0.009634283371269703, 0.010015774518251419,
            0.01641061156988144,
            -0.0029694929253309965, -0.006362506654113531, 0.0022711690980941057, 0.03339022770524025,
            -0.009052346460521221, -0.016643386334180832, 0.005211565177887678, -0.0033526013139635324,
            -0.015996789559721947, 0.004354825243353844, 0.0027415677905082703, -0.008948891423642635,
            0.0025023273192346096, -0.006970306858420372, -0.011735720559954643, -0.00597778195515275,
            0.018518514931201935, 0.016759773716330528, -0.0029565610457211733, -0.010979203507304192,
            -0.0041414485312998295, -0.022113589569926262, 0.003053550375625491, -5.049007086199708e-05,
            -0.0042966315522789955, 0.024156833067536354, 0.017742598429322243, 0.007228945381939411,
            0.010300276800990105,
            0.005680347327142954, 0.016695113852620125, -0.026316463947296143, -0.008463944308459759,
            0.012576295994222164,
            0.019630659371614456, -0.00019812110986094922, 0.006259051151573658, 0.02829504944384098,
            0.012485772371292114,
            -0.015789879485964775, 0.02537243440747261, -0.01699254848062992, 0.02738981507718563,
            -0.0062849149107933044,
            0.005867860279977322, 0.019203906878829002, 0.003591841785237193, -0.004907664842903614,
            -0.09321330487728119,
            -0.004594065714627504, -0.007015568669885397, 0.039856187999248505, -0.028036409988999367,
            0.007461720146238804, -0.009983445517718792, 0.00902001652866602, 0.018932336941361427,
            0.030959025025367737,
            -0.002788446145132184, -0.038019854575395584, -0.02551468461751938, 0.0004994955379515886,
            0.004920597188174725, 0.01513035036623478, 0.005415243096649647, -0.03685598447918892,
            0.00223560631275177,
            0.01568642258644104, -0.018867677077651024, 0.0189581997692585, 0.002177412621676922,
            -0.007222479209303856,
            0.002406954299658537, -0.00922046136111021, -0.03918372839689255, 0.012479306198656559,
            0.005667415447533131,
            0.008321693167090416, 0.02100144326686859, 0.005554261151701212, 0.004813908599317074,
            -0.015841607004404068,
            -0.033286772668361664, 0.009252791292965412, 0.006504757795482874, 0.009407974779605865,
            0.009621351025998592,
            -0.03628697618842125, -0.0014847464626654983, -0.018686629831790924, 0.0031392243690788746,
            -0.02426028810441494, 0.004982023499906063, 0.02808813750743866, -0.013643179088830948,
            -0.01585453934967518,
            0.014367367140948772, -0.020122073590755463, -0.004193176049739122, -0.016113176941871643,
            -0.03359713777899742, 0.016708046197891235, 0.00702850054949522, -0.01130250096321106,
            0.001401497283950448,
            0.017742598429322243, 0.005544562358409166, -0.014716529287397861, -0.020147936418652534,
            0.01753568835556507,
            -0.01004163920879364, 0.012556897476315498, 0.0015582968480885029, 0.01825987547636032,
            -0.02154458500444889,
            -0.03103661723434925, 0.004206108395010233, -0.015505376271903515, 0.01086928229779005,
            0.0070349667221307755,
            -0.019074587151408195, 0.015427784994244576, -0.027053585276007652, 0.007694494444876909,
            -0.009989910759031773, -0.02702772058546543, 0.0020561758428812027, -0.014444958418607712,
            -0.01585453934967518, -0.027958819642663002, 0.004949694033712149, -0.019087519496679306,
            -0.009453236125409603, 0.0251784548163414, -0.007086694240570068, -0.0008025875431485474,
            -0.006572650279849768, -0.031269390136003494, 0.015195010229945183, 0.0044227177277207375,
            -0.0004214998916722834, -0.0013045078376308084, -0.011347763240337372, 0.014936371706426144,
            -0.004170545376837254, -0.0016205316642299294, 0.0057579390704631805, -0.00886483397334814,
            -0.0075781075283885, -0.007688028737902641, -0.04249430075287819, 0.025450026616454124,
            0.017846055328845978,
            -0.011380093172192574, -0.00866438914090395, -0.014832916669547558, -0.01330694928765297,
            0.006527388468384743,
            -0.0044518145732581615, 0.0020594089291989803, -0.003391396952793002, 0.013565587811172009,
            -0.025825051590800285, -0.0009755520150065422, -0.012097815051674843, -0.028708871454000473,
            0.010183890350162983, 0.007241877261549234, 0.0017862219829112291, -0.007739756256341934,
            -0.018376262858510017, 0.007177217863500118, -0.011192579753696918, 0.001428169314749539,
            -0.02116956003010273,
            -0.0007152970065362751, -0.012291793711483479, 0.04585660248994827, -0.010862816125154495,
            0.0014435260090976954, 0.008696719072759151, -0.025954371318221092, 0.012304725125432014,
            0.014276843518018723,
            0.0028337077237665653, -0.01937202177941799, 0.01935908943414688, -0.010714098811149597,
            0.019307361915707588,
            -0.03173493966460228, -0.000870480143930763, -0.02098851278424263, 0.0209497157484293,
            0.006116800010204315,
            -0.011160249821841717, -0.011386558413505554, 0.009983445517718792, -0.007002636790275574,
            0.007597505114972591, -0.009039415046572685, 0.024143900722265244, 0.02406631037592888,
            0.006178226787596941,
            -0.02369128353893757, -0.006107100751250982, -0.0005431408062577248, 0.011561139486730099,
            0.005732075311243534, 0.0032426798716187477, -0.03525242581963539, 0.0067504639737308025,
            0.00014134187949821353, 0.015453648753464222, -0.012615091167390347, 0.01697961613535881,
            0.009207529947161674,
            -0.011205512098968029, -0.0012883428717032075, 0.021079035475850105, -0.022579139098525047,
            -0.027234630659222603, -0.01324229035526514, 0.017212390899658203, -0.004170545376837254,
            0.018518514931201935,
            -0.010332606732845306, 0.020755738019943237, 0.0105977114289999, -0.02629060111939907,
            0.02992447093129158,
            0.013345745392143726, 0.007726824376732111, -0.009576089680194855, -0.018376262858510017,
            0.02244981937110424,
            -0.0097506707534194, -0.017923645675182343, 0.0065888152457773685, 0.002591234166175127,
            0.018117625266313553,
            -0.031062480062246323, 0.002280868124216795, -0.013591451570391655, 0.0033202713821083307,
            -0.025437094271183014, 0.048106756061315536, -0.009711874648928642, -0.001504144398495555,
            0.015518308617174625, -0.010694701224565506, -0.0037373260129243135, -0.006417467258870602,
            -0.019397886469960213, -0.007985463365912437, -0.023290393874049187, -0.001201052451506257,
            -0.019449613988399506, -0.01972118392586708, -0.00903294887393713, 0.0018848278559744358,
            0.02427322044968605,
            -0.018932336941361427, 0.00026631681248545647, 0.03504551202058792, -0.01404406875371933,
            0.009485566057264805,
            -0.01604851707816124, -0.02588971145451069, -0.017949510365724564, 0.015350193716585636,
            0.011005067266523838,
            0.011580538004636765, 0.02663976326584816, -0.019992753863334656, 0.006802191957831383,
            -0.004865636117756367,
            0.010461926460266113, -0.010617109015583992, 0.009136403910815716, -0.011910301633179188,
            -0.00561245484277606,
            0.015272601507604122, -0.00432572839781642, 0.0019009928219020367, -0.00949203222990036,
            0.0005059615359641612,
            0.008767844177782536, -0.002914532320573926, -0.028165729716420174, 0.044434089213609695,
            -0.008198839612305164, -0.00033057230757549405, 0.015996789559721947, 0.0007904638187028468,
            0.015505376271903515, 0.006802191957831383, 0.006912113167345524, -0.015440717339515686,
            -0.017018411308526993,
            0.0027367183938622475, 0.002512026112526655, 0.017600348219275475, -0.014936371706426144,
            -0.008425148203969002, 0.005321486387401819, -0.0019995986949652433, -0.0006785218720324337,
            -0.03380404785275459, -0.0042610689997673035, 0.027984682470560074, 0.017975373193621635,
            0.014328571036458015,
            0.019268566742539406, -0.014626005664467812, 0.00865145679563284, 0.013054776936769485,
            -0.009705408476293087,
            -0.013979409821331501, -0.029303738847374916, 0.0027852130588144064, 0.01176158431917429,
            -0.01841505989432335,
            -0.030984889715909958, 0.021130762994289398, -0.004219040274620056, -0.003950702492147684,
            0.004270767793059349, 0.010849883779883385, 0.016850296407938004, -0.016320087015628815,
            0.0390026830136776,
            -0.03398509696125984, -0.013080640695989132, -0.011742186732590199, -0.009550225920975208,
            -0.008237635716795921, 0.00013093573215883225, -0.008793707937002182]
    # Security discovered a concealed knife in the passenger's bag.
    arr = [0.01877044141292572, 0.007516102399677038, 0.024635378271341324, -0.017475929111242294,
           -0.016828671097755432, 0.019364861771464348, -0.025877054780721664, -0.020276304334402084,
           -0.003586329985409975, -0.0037019115407019854, 0.03201938793063164, 0.008328475058078766,
           -0.008361498825252056, 0.005372890271246433, -0.003956190776079893, -9.958587907021865e-05,
           0.02631296217441559, 0.004517586901783943, 0.030090827494859695, -0.03965437412261963, -0.005118610803037882,
           0.005749356001615524, -0.01251582894474268, -0.006789589766412973, -0.02253509685397148, 0.01142606046050787,
           0.004682703409343958, -0.025969520211219788, 0.009761686436831951, -0.020870722830295563,
           0.00572954211384058, -0.00238923542201519, -0.03846553713083267, -0.006654194090515375,
           -0.027950918301939964, -0.005244099535048008, 0.0117562934756279, -0.022878538817167282,
           0.002235677093267441, 0.007727451156824827, 0.019998908042907715, -0.012489411048591137,
           -0.004279819317162037, 0.006634380202740431, -0.015811555087566376, 0.023631470277905464,
           -0.00900215096771717, 0.007218892686069012, -0.028400035575032234, 0.027290452271699905, 0.01755518466234207,
           0.0033502134028822184, -0.03381585702300072, -0.014279273338615894, 0.009986245073378086,
           -0.012846061959862709, 0.004943587351590395, 0.009226708672940731, 0.0019269093172624707,
           -0.029879478737711906, -0.009992849081754684, -0.01109582744538784, -0.01686829887330532,
           0.010897687636315823, 0.00030525910551659763, -0.007872753776609898, 0.011743084527552128,
           0.028717057779431343, -0.01904783770442009, 0.013816947117447853, -0.0006175356102176011,
           0.011333595030009747, 0.018268488347530365, -0.011736479587852955, 0.017925044521689415,
           -0.005138424690812826, -0.012258247472345829, 0.0011549898190423846, 0.009794709272682667,
           0.016049321740865707, 0.012476201169192791, -0.03172878175973892, -0.01721174269914627, 0.02633938193321228,
           0.019932862371206284, -0.03281194716691971, -0.01973472163081169, 0.01748913712799549, -0.009068197570741177,
           -0.0017964673461392522, 0.0016041066264733672, 0.01000605896115303, -0.031543850898742676,
           0.00032094516791403294, -0.023697517812252045, 0.029060499742627144, -0.022601144388318062,
           0.015494531020522118, -0.004408610053360462, -0.0271319393068552, 0.011657223105430603, 0.006855636369436979,
           -0.02813584916293621, -0.0011500363470986485, -0.024556122720241547, -0.027765987440943718,
           -0.013057411648333073, 0.0007450880948454142, -0.0032544457353651524, -0.018889324739575386,
           -0.004151028115302324, -0.00017285632202401757, 0.00885024294257164, -0.02364468015730381,
           -0.016643742099404335, 0.017158905044198036, 0.00029163697035983205, -0.02426551841199398,
           0.0007310532382689416, -0.03532171621918678, 0.017304208129644394, 0.011076013557612896,
           0.016551276668906212, -0.0039165630005300045, 0.02080467715859413, 0.014979367144405842,
           -0.015917228534817696, -0.006849031429737806, 0.010937315411865711, -0.007661404553800821,
           0.013361225835978985, -0.0018542581237852573, 0.022442631423473358, -0.0006266170530579984,
           -0.015917228534817696, 0.015045413747429848, -0.02789808064699173, 0.021095281466841698,
           -0.016221042722463608, -0.02682812511920929, 0.007767079398036003, 0.02094997838139534, -0.04359075054526329,
           0.014913320541381836, 0.0008292975253425539, 0.010078709572553635, 0.0014455948257818818, 0.0356123223900795,
           -0.010461780242621899, -0.004418516997247934, 0.01881006918847561, -0.013592388480901718,
           0.01346690021455288, -0.027079103514552116, -0.021768957376480103, 0.015124669298529625, 0.01221861969679594,
           -0.014583087526261806, 0.0012870830250903964, -0.011109036393463612, -0.02485993690788746,
           0.014569878578186035, 0.02530905418097973, -0.0190082099288702, 0.019853604957461357, 0.02855854667723179,
           -0.0028466081712394953, -0.0035632136277854443, 0.006416426505893469, 0.0071660554967820644,
           0.0026666312478482723, -0.008810615167021751, -0.014530249871313572, 0.00826242845505476,
           0.02595631033182144, 0.011531734839081764, 0.021517978981137276, -0.011135455220937729,
           -0.039152421057224274, -0.001351478393189609, -0.02323519065976143, 0.01838737167418003,
           0.024846728891134262, 0.025071285665035248, -0.0038934466429054737, -0.003834004746749997,
           0.03048710711300373, 0.017198532819747925, -0.001367164426483214, -0.013414062559604645,
           0.017713695764541626, 0.017343835905194283, -0.01130717620253563, -0.02035555988550186, -0.6725128293037415,
           -0.026669614017009735, 0.01808355748653412, -0.004927075933665037, 0.017277788370847702, 0.02364468015730381,
           0.021280212327837944, 0.022495469078421593, -0.019206348806619644, 0.015811555087566376,
           0.0008875011117197573, 0.028320778161287308, 0.008757778443396091, -0.002607189118862152,
           -0.002232374856248498, -0.009807919152081013, -0.005960704758763313, -0.0006039135041646659,
           0.02336728386580944, 0.029192592948675156, -0.01763444021344185, 0.023565424606204033, -0.005204471293836832,
           0.016221042722463608, 0.022455841302871704, 0.0022389795631170273, 0.007621776778250933,
           -0.02589026466012001, -0.0010988501599058509, -0.0044350288808345795, -0.01450383197516203,
           -0.002070560585707426, 0.007225497160106897, 0.008955918252468109, 0.04887447878718376, -0.02022346667945385,
           0.014081133529543877, 0.024503285065293312, 0.013909412547945976, 0.014477413147687912,
           -0.0019302116706967354, -0.006366891320794821, 0.00793219543993473, -0.01112885121256113,
           -0.008671917952597141, 0.0020722118206322193, 0.04691949859261513, -0.008169963024556637,
           0.006713636219501495, -0.00829545222222805, 0.027396125718951225, 0.010455175302922726, 0.007945405319333076,
           0.0006092798430472612, -0.012661132030189037, -0.0018740720115602016, 0.01949695497751236,
           -0.03141175955533981, 0.011300572194159031, -0.022165236994624138, -0.009068197570741177,
           0.02858496457338333, -0.008315266110002995, -0.01836095191538334, -0.008308661170303822,
           0.020276304334402084, -0.011591176502406597, -0.0013803737238049507, 0.01301778294146061,
           -0.007985033094882965, 0.020064955577254295, 0.029377523809671402, -0.013935831375420094,
           -0.0020391885191202164, -0.00044705288019031286, 0.036589812487363815, 0.026048775762319565,
           -0.010712757706642151, -0.015283181332051754, 0.002589026466012001, -0.003969400189816952,
           -0.00014179377467371523, -0.03014366514980793, -0.00022270085173659027, 0.037699393928050995,
           -2.369679532421287e-05, 0.002374375006183982, -0.004534098785370588, 0.013400853611528873,
           -0.008586057461798191, 0.0021101885940879583, 0.017132485285401344, -0.011498712003231049,
           0.0019962582737207413, -0.008057684637606144, 0.03104189783334732, 0.006360286846756935,
           0.003814190626144409, 0.01870439574122429, -0.008645499125123024, 0.00510870385915041,
           -0.0037811673246324062, 0.026920590549707413, -0.026048775762319565, 0.01130717620253563,
           0.00017595225654076785, -0.0020358862821012735, -0.01342727243900299, 0.0034839578438550234,
           -0.02430514618754387, 0.005148332100361586, 0.012766806408762932, -0.00700093898922205, -0.02295779623091221,
           -0.023723935708403587, -0.02381640113890171, 0.015111460350453854, 0.019972490146756172,
           0.007284939289093018, -0.04998406022787094, 0.011082618497312069, 0.00253618904389441, 0.01134019996970892,
           -0.012581875547766685, 0.015124669298529625, 0.026920590549707413, -0.00457042409107089,
           -0.02606198564171791, -0.013037597760558128, -0.014701971784234047, -0.010864664800465107,
           0.00029597128741443157, 0.025903472676873207, 0.0018674674211069942, -0.0012193851871415973,
           -0.008103916421532631, 0.004983215592801571, -0.013750900514423847, 0.003725027898326516,
           -0.01881006918847561, -0.03777864947915077, -0.027264032512903214, -0.015322809107601643,
           -0.005580937024205923, 0.002906050067394972, -0.032442085444927216, -0.00954373273998499,
           0.0020342350471764803, -0.024661798030138016, -0.009510708972811699, 0.019245976582169533,
           -0.004230284132063389, -0.01229127123951912, -0.030672037973999977, 0.021623654291033745,
           -0.0020127699244767427, -0.011135455220937729, 0.0014010133454576135, -0.002977050142362714,
           -0.0003750208124984056, 0.00896912720054388, 0.03135892376303673, -0.03196655213832855,
           -0.00022435201390180737, -0.019589418545365334, 0.011498712003231049, 0.049587782472372055,
           0.014543459750711918, -0.03999781608581543, -0.04639112576842308, 4.378166704555042e-05,
           -0.02191425859928131, 0.010844850912690163, 0.0006179484189487994, -0.000650558911729604,
           -0.0007182566914707422, -0.007971824146807194, -0.0016255717491731048, 0.02005174569785595,
           0.000143960933201015, -0.0012771759647876024, -0.0075689395889639854, -0.023354075849056244,
           -0.007661404553800821, 0.010659920051693916, 0.015098251402378082, 0.011822340078651905, 0.03270627185702324,
           -0.013869784772396088, 0.014094342477619648, 0.010745780542492867, -0.017581602558493614, 0.0169343464076519,
           0.007423636969178915, -0.03003798983991146, 0.025084495544433594, 0.01960262842476368, 0.020064955577254295,
           0.00427651684731245, 0.01752876490354538, 0.016432391479611397, -0.004170842468738556, -0.010243826545774937,
           -0.0025857239961624146, 0.006102704908698797, -0.03640488162636757, -0.0046529825776815414,
           -0.032785527408123016, 0.040156327188014984, 0.014398157596588135, 0.022548306733369827,
           -0.02039518766105175, 0.00490065710619092, -0.023842819035053253, -0.00648907758295536, 0.005680006928741932,
           -0.01701360195875168, 0.014398157596588135, 0.009464476257562637, -0.0027557939756661654,
           0.017330626025795937, 0.005187959875911474, 0.026537520810961723, -0.0020639561116695404,
           -0.005353076383471489, 0.014781227335333824, 0.022653982043266296, 0.014186807908117771,
           0.011379827745258808, -0.030539944767951965, -0.0038538186345249414, 0.020144211128354073,
           0.01005889568477869, 0.014107552357017994, 0.012093131430447102, 0.003827400039881468, 0.014834064990282059,
           -0.010706152766942978, 0.032996878027915955, -0.010831641033291817, 0.00433926098048687,
           0.010019267909228802, 0.03574441745877266, -0.008473778143525124, 0.02250867895781994, 0.0020111186895519495,
           0.044753171503543854, -0.0009353848872706294, -0.007357590366154909, 0.009821128100156784,
           0.02028951235115528, -0.0015116414288058877, 0.015586995519697666, 0.0038967488799244165,
           0.008011451922357082, -0.016366345807909966, 0.034000784158706665, 0.013275365345180035,
           0.026973428204655647, 0.02956245467066765, 0.002169630490243435, 0.005663495510816574, 0.007952009327709675,
           -0.005825309548527002, -0.0050558666698634624, -0.00037976790918037295, 0.005699820816516876,
           -0.005491774063557386, -0.0043260520324110985, 0.017911836504936218, 0.006888659670948982,
           -0.030883386731147766, 0.0162474624812603, -0.007595357950776815, 0.01956300064921379, 0.009847546927630901,
           0.005977216642349958, 0.006122519262135029, 0.02133304998278618, -0.007701032795011997,
           -0.020580118522047997, -0.047342196106910706, 0.0012012224178761244, 0.023420121520757675,
           -0.013935831375420094, -0.01669657789170742, -0.004801587201654911, -0.0016924439696595073,
           -0.021161329001188278, 0.0245429128408432, 0.0016536415787413716, 0.038148511201143265,
           -0.012984760105609894, -0.019298814237117767, -0.003333701752126217, -0.034793343394994736,
           0.02184821292757988, -0.017330626025795937, 0.0023974913638085127, 0.009319174103438854,
           -0.012859271839261055, -0.029985152184963226, -0.040499769151210785, -0.01603611186146736,
           0.031226828694343567, 0.005158239044249058, -0.009239918552339077, -0.022495469078421593,
           0.02046123519539833, 0.010534431785345078, 0.021267002448439598, -0.0007591230096295476,
           -0.00918047595769167, -0.015058622695505619, -0.007331171538680792, 0.01956300064921379, -0.0119280144572258,
           0.010309873148798943, 0.018968582153320312, 0.0026418636552989483, -0.020170629024505615,
           -0.01656448468565941, -0.026947010308504105, -0.0008359021740034223, 0.10514617711305618,
           0.016987184062600136, -0.006746659521013498, 0.002816887106746435, 0.015785135328769684,
           -0.01943090744316578, -0.03349883109331131, 0.020197048783302307, 0.005455448757857084,
           -0.024278726428747177, -0.010362710803747177, 0.0015875949757173657, 0.0071660554967820644,
           -0.005260610952973366, 0.02606198564171791, -0.002451979788020253, -0.01714569516479969,
           0.021993516013026237, 0.0038769349921494722, -0.03912600129842758, -0.0004895703750662506,
           -0.01797788217663765, 0.006413124036043882, 0.01196103822439909, -0.027290452271699905,
           -0.034344229847192764, 0.015785135328769684, 0.0171060673892498, 0.01849304512143135, 0.012350712902843952,
           0.011564758606255054, 0.020580118522047997, 0.012661132030189037, -0.005234192591160536,
           -0.024529704824090004, -0.009933407418429852, 0.0005081459530629218, -0.01239034067839384,
           -0.0067169382236897945, -0.00015324873675126582, 0.022878538817167282, -0.006485775578767061,
           0.019549790769815445, -0.004035446792840958, 0.01759481243789196, -0.004610052332282066, 0.01552094891667366,
           0.018928952515125275, -0.004828006029129028, -0.015534158796072006, 0.014834064990282059,
           0.012212014757096767, -0.010382524691522121, 0.0001750234659994021, 0.03003798983991146,
           0.021425515413284302, 1.9684979633893818e-05, 0.0024569332599639893, -0.028162267059087753,
           -2.5618854124331847e-05, -0.00938522070646286, -0.017621230334043503, 0.009345592930912971,
           -0.01652485691010952, -0.022653982043266296, -0.016828671097755432, -0.00623149611055851,
           -0.013790528289973736, -0.021597236394882202, -0.003006771206855774, 0.018136395141482353,
           -0.009682430885732174, 0.0005667623481713235, 0.0037019115407019854, 0.009827733039855957,
           0.004332656506448984, 0.013176294974982738, -0.009444662369787693, 0.01030326820909977, 0.009134244173765182,
           0.009147453121840954, -0.010659920051693916, 0.0025626078713685274, -0.026299752295017242,
           0.003338655224069953, 0.015203925780951977, 0.002062304876744747, -0.010151361115276814,
           -0.023935284465551376, 0.007119822781533003, 0.03637846186757088, 0.0007471520802937448,
           0.006835822481662035, -0.014450994320213795, -0.0015479669673368335, -0.016604112461209297,
           -0.009959826245903969, 0.033683761954307556, 0.013367829844355583, 0.011439269408583641,
           0.035004694014787674, -0.013295179232954979, 0.005577634554356337, -0.010798618197441101,
           0.018955372273921967, 0.00019421825709287077, -0.005580937024205923, -0.00812373124063015,
           0.02229733020067215, 0.034344229847192764, -0.0016239206306636333, -0.01780616119503975,
           -0.02039518766105175, -0.00496009923517704, 0.007575544063001871, 0.00918047595769167, -0.00587814673781395,
           0.005158239044249058, 0.015468112193048, -0.02544114738702774, 0.008513405919075012, 0.003576423041522503,
           0.01583797298371792, 0.012159178033471107, 0.022653982043266296, 0.0075689395889639854, 0.00912763923406601,
           -0.020857512950897217, -0.02879631519317627, 0.010996758006513119, 0.0021977003198117018,
           0.0023297935258597136, -0.024741053581237793, 0.0037514464929699898, -0.000504843657836318,
           -0.01921955868601799, 0.0022637469228357077, 0.012542247772216797, -0.013803738169372082,
           -0.011419455520808697, -0.02710552141070366, -0.004055260680615902, -0.001493478543125093,
           -0.0037844697944819927, -0.023789983242750168, -0.01367164496332407, -0.015045413747429848,
           0.021214164793491364, -0.02198030613362789, 0.015996484085917473, -0.0027145149651914835, 0.0108052222058177,
           -0.018625138327479362, -0.007575544063001871, -0.021768957376480103, -0.0414772592484951,
           -0.021002816036343575, 0.01590401865541935, 0.01742309145629406, 0.03217789903283119, 0.0349254384636879,
           -0.011492107063531876, 0.026643196120858192, 0.025071285665035248, -0.006439542863518, 0.0018014208180829883,
           -0.00909461546689272, 0.003373329760506749, -0.014133971184492111, 0.024173052981495857,
           0.028373615816235542, -0.009484291076660156, 0.004933680407702923, 0.01949695497751236,
           -0.026365799829363823, 0.006994334049522877, 0.009695639833807945, 0.010937315411865711,
           -0.006158844567835331, -0.008658708073198795, -0.010792013257741928, -0.031121155247092247,
           0.0023149331100285053, 0.025916682556271553, -0.06409161537885666, -0.0045869359746575356,
           0.0065386127680540085, 0.005287029780447483, 0.011888386681675911, -0.012984760105609894,
           0.03408004343509674, -0.02188784070312977, -0.012542247772216797, 0.002792119747027755,
           -9.287802095059305e-05, 0.001885630190372467, -0.012264852412045002, 0.01863834820687771,
           0.004203865770250559, 0.024661798030138016, 0.02295779623091221, 0.0072585204616189, -0.006436240393668413,
           -0.030619200319051743, 0.007192473858594894, 0.03048710711300373, -0.006300844717770815,
           -0.013011178933084011, 0.009583360515534878, -0.013189504854381084, -0.023552214726805687,
           -0.023671098053455353, -0.043432239443063736, -0.02993231639266014, -0.006459356751292944,
           0.0038934466429054737, -0.01355276070535183, 0.017713695764541626, -0.02544114738702774,
           -0.013030992820858955, -0.011888386681675911, 0.021874630823731422, 0.03341957554221153,
           -0.02405416965484619, 0.03651055693626404, 0.013281969353556633, -0.01163080520927906, -0.018268488347530365,
           0.03402720391750336, 0.015745507553219795, -0.017264578491449356, 0.008876661770045757, 0.025797799229621887,
           0.005303541198372841, -0.006555124185979366, -0.020421605557203293, 0.016128577291965485,
           -0.008051079697906971, -0.02291816845536232, 0.007238706573843956, 0.019061045721173286,
           0.011624200269579887, -0.02212560921907425, -0.025005239993333817, -0.02098960615694523,
           0.0007471520802937448, -0.005831914022564888, 0.0160757414996624, 0.017436299473047256,
           -0.013004573993384838, -0.010811827145516872, 0.010428756475448608, -0.016947556287050247,
           0.026458265259861946, 0.008202986791729927, -0.012608294375240803, 0.0026451661251485348,
           -0.020131001248955727, 0.007621776778250933, -0.014807646162807941, -0.013486714102327824,
           0.009563546627759933, 0.016749415546655655, 8.813092426862568e-05, 0.00507237808778882,
           -0.007199078798294067, -0.033234644681215286, 0.006822613067924976, 0.0024239099584519863,
           0.02423909865319729, -0.0016627229051664472, -0.020593328401446342, -0.02578458935022354,
           -0.006594752427190542, -0.015653042122721672, 0.0188364889472723, 0.0006237275083549321,
           -0.022376585751771927, 0.0073443809524178505, -0.009160662069916725, 0.005746053531765938,
           0.02423909865319729, -0.013711272738873959, 0.0002767764963209629, -0.03204580768942833,
           -0.014358528889715672, -0.02039518766105175, -0.02392207644879818, 0.016881508752703667,
           -0.005471960175782442, -0.04131874814629555, -0.004408610053360462, -0.007364194840192795,
           0.015996484085917473, -0.00392316747456789, -0.010970339179039001, 0.01828169636428356, 0.044885262846946716,
           -0.015877600759267807, 0.011776107363402843, -0.01648522913455963, 0.03225715458393097,
           -0.015111460350453854, -0.0004581982211675495, 0.011974247172474861, -0.02088393270969391,
           0.04440972954034805, -0.022772865369915962, -7.574718620162457e-05, -0.016722997650504112,
           -0.02268039993941784, 0.011762898415327072, 0.014371738769114017, -0.015005785971879959,
           0.009226708672940731, 0.00044787846854887903, 0.002019374631345272, 0.0011904898565262556,
           -0.011379827745258808, 0.0188364889472723, -0.043088797479867935, 0.0053563788533210754,
           0.018625138327479362, -0.004487866070121527, -0.02640542760491371, 0.01013154722750187, 0.01313666719943285,
           0.0052077737636864185, -0.022455841302871704, -0.011690246872603893, -0.003302329685539007,
           0.025388309732079506, 0.011452479287981987, -0.017753323540091515, 0.0023859331849962473,
           0.019312024116516113, -0.021901050582528114, -0.005762565415352583, 0.007139636669307947,
           -0.002189444610849023, 0.02392207644879818, -0.025427937507629395, -0.009286151267588139,
           0.007245311047881842, -0.015705879777669907, -0.0077538699842989445, 0.006449449807405472,
           -0.021042443811893463, -0.010778804309666157, 0.004557214677333832, -0.028875570744276047,
           0.02188784070312977, -0.007086799480021, -0.016815463081002235, -0.009682430885732174, -0.027528218924999237,
           -0.01915351115167141, 0.005749356001615524, -0.023327656090259552, 0.02758105657994747, 0.02375035546720028,
           0.0152963912114501, -0.005491774063557386, 0.02813584916293621, 0.009959826245903969, 0.014820855110883713,
           0.01853267289698124, -0.008070893585681915, 0.027184776961803436, -0.001966537209227681,
           0.00036036671372130513, 0.030328596010804176, -0.013281969353556633, -0.006373496260493994,
           -0.030962642282247543, -0.00433926098048687, 0.008269033394753933, 0.0014794436283409595,
           -0.0214387234300375, -0.011644014157354832, 0.008770987391471863, 0.0007574718329124153, 0.02492598444223404,
           0.012139363214373589, -0.009821128100156784, -0.021729329600930214, 6.836854299763218e-05,
           0.0007752218516543508, 0.027871662750840187, -0.00900215096771717, -0.0005886402796022594,
           0.021412305533885956, 0.0021200955379754305, 0.005227587651461363, -0.01828169636428356,
           0.009616384282708168, 0.0217821653932333, -0.02533547207713127, 0.007549125701189041, -0.007516102399677038,
           0.008024660870432854, 0.019800769165158272, 0.00533986696973443, 0.00834828894585371, -0.02744896337389946,
           0.00929936021566391, -0.01977434940636158, -0.004316144622862339, 0.020091373473405838,
           -0.026524310931563377, 0.032415665686130524, -0.018519464880228043, 0.007232101634144783,
           0.0013745947508141398, 0.0036490741185843945, 0.007621776778250933, -0.014067924581468105,
           -0.029192592948675156, 0.02167649194598198, -0.022046351805329323, -0.015085041522979736,
           0.008995546028017998, 0.0007735707331448793, -0.019470535218715668, 0.032336410135030746,
           -0.009411639533936977, 0.005594146437942982, -0.008500196039676666, -0.001761792809702456,
           0.0024569332599639893, 0.00040866329800337553, -0.01887611672282219, 0.016709787771105766,
           -0.004709122236818075, -0.017515556886792183, 0.00032156435190699995, 0.22001440823078156,
           -0.0027623986825346947, -0.003728330135345459, 0.018572302535176277, 0.005749356001615524,
           -0.006948101334273815, 0.02270681783556938, -7.440561603289098e-05, -0.004920470993965864,
           0.01697397418320179, -0.009761686436831951, 0.004418516997247934, 0.004861029330641031,
           -0.002493258798494935, 0.005620564799755812, 0.015983276069164276, -0.018545882776379585,
           -0.012205409817397594, -0.026048775762319565, -0.012628108263015747, -0.0006728496518917382,
           0.007456660270690918, 0.018096765503287315, -0.006915078032761812, 0.014715180732309818,
           0.011980852112174034, 0.002475096145644784, 0.005571030080318451, 0.01648522913455963, -0.002843305701389909,
           0.003062910633161664, -0.007747265510261059, 0.028347197920084, -0.015851182863116264, -0.003434422891587019,
           0.008460568264126778, -0.003201608546078205, 0.012443178333342075, -0.005452146288007498,
           0.004705819766968489, -0.00938522070646286, 0.002204305026680231, 0.028320778161287308,
           -0.009438058361411095, -0.03756730258464813, -0.007040566764771938, -0.013830156065523624,
           -0.016102159395813942, 0.0023644680622965097, -0.009596569463610649, -0.01517750695347786,
           -0.013354620896279812, 0.0073509858921170235, 0.007403823081403971, -0.0188364889472723,
           0.005997030530124903, 0.005597448907792568, -0.006181960925459862, 0.01450383197516203,
           -0.013519737869501114, 0.004715726710855961, -0.006000332999974489, -0.023618262261152267,
           0.018744023516774178, -0.012119549326598644, 0.03719744086265564, -0.010521221905946732, 0.03772581368684769,
           0.010620292276144028, -0.03186087682843208, 0.007601962890475988, 0.007020752876996994,
           -0.013367829844355583, -0.0010798617731779814, -0.02077825739979744, -0.020315932109951973,
           0.015005785971879959, -0.01071936171501875, 0.009438058361411095, 0.010151361115276814,
           -0.0002008229203056544, -0.011287362314760685, 0.019351651892066002, -0.004953494295477867,
           -0.0033254458103328943, -0.03481976315379143, -0.004811494145542383, 0.007080194540321827,
           0.014834064990282059, -0.03376301750540733, 0.003039794508367777, -0.03471408784389496,
           -0.0032313296105712652, -0.02890198864042759, 0.0055016810074448586, 0.02475426346063614,
           0.012027084827423096, 0.03072487562894821, 0.00012579812027979642, 0.017026811838150024,
           -0.024397611618041992, -0.004970006179064512, 0.015507739968597889, -0.03035501390695572,
           -0.016789043322205544, 0.026352589949965477, -0.016128577291965485, 0.023776773363351822,
           0.02001211792230606, -0.03497827425599098, -0.002318235347047448, -0.03606143966317177,
           0.0046133543364703655, 0.0014967808965593576, -0.011954433284699917, 0.0080378707498312,
           0.019787559285759926, -0.0053497739136219025, -0.006300844717770815, -0.025507193058729172,
           0.02153118886053562, -0.003903353586792946, 0.016128577291965485, -0.0004540703084785491,
           0.003295724978670478, 0.0021910956129431725, 0.0020111186895519495, -0.00980131421238184,
           0.009457872249186039, -0.024173052981495857, 0.01669657789170742, -0.0013473505387082696,
           0.011525130830705166, 0.0028614685870707035, 0.008698335848748684, -0.008665313012897968,
           0.01680225320160389, 0.008427545428276062, 0.018677975982427597, -0.013334807008504868, 0.03685399889945984,
           0.005772472359240055, -0.016405973583459854, -0.014530249871313572, 0.028426453471183777,
           -0.013262155465781689, 0.028743477538228035, -0.00812373124063015, -0.0140018779784441,
           -0.019457325339317322, -0.021425515413284302, -0.014160389080643654, -0.02758105657994747,
           -0.03162311017513275, 0.006730147637426853, -0.018519464880228043, -0.034000784158706665,
           -0.02305026166141033, 0.015402065590023994, 0.012093131430447102, -0.0062447055242955685,
           4.273033937352011e-06, 0.014926529489457607, -0.007694427855312824, 0.014715180732309818,
           -0.006360286846756935, -0.16960765421390533, 0.03083054907619953, 0.0009469430078752339,
           -0.03463483229279518, -0.004738843068480492, 0.026709241792559624, 0.0006637682672590017,
           0.010118338279426098, -0.01855909265577793, 0.008982336148619652, 0.024173052981495857,
           -0.004359074868261814, -0.03141175955533981, 0.0005944193108007312, -0.005475262645632029,
           0.00137211789842695, -0.010673128999769688, 0.02281249314546585, 0.009292755275964737, 0.009008754976093769,
           0.029166175052523613, -0.018479837104678154, 0.0264450553804636, -0.01552094891667366,
           -0.0013927575200796127, -0.008777592331171036, -0.019166721031069756, 0.023393703624606133,
           -0.025322264060378075, -0.003903353586792946, 5.143765520188026e-06, 0.006967915687710047,
           0.0073509858921170235, 0.03347241133451462, 0.019417697563767433, -0.016617322340607643,
           0.0014571528881788254, -0.013803738169372082, -0.0002782212686724961, 0.023803191259503365,
           0.0016140135703608394, 0.038491953164339066, 0.020751839503645897, 0.0010228966129943728,
           0.004933680407702923, 0.020553698763251305, 0.011551548726856709, 0.011888386681675911, 0.00590126309543848,
           -0.0050558666698634624, 0.008929499424993992, -0.021650072187185287, 0.002280258573591709,
           0.0017799556953832507, 0.008427545428276062, 0.026128031313419342, 0.009940012358129025,
           -0.007443450856953859, 0.006515496410429478, -0.001655292697250843, 0.0006633554585278034,
           -0.02077825739979744, -0.012846061959862709, -0.005930983927100897, -0.025388309732079506,
           -0.01818923093378544, -0.00034674460766837, -0.016683369874954224, 0.0015165949007496238,
           -0.003001817734912038, -0.012766806408762932, 0.01877044141292572, 0.01718532294034958,
           -0.008982336148619652, 0.006310752127319574, 0.012819643132388592, 0.003596236929297447, 0.02530905418097973,
           -0.0063536823727190495, -0.015666252002120018, -0.006882054731249809, 0.043405819684267044,
           0.005105401389300823, -0.013519737869501114, -0.007621776778250933, 0.018955372273921967,
           -0.005557820666581392, -0.009840941987931728, 0.003936376888304949, 0.0029176082462072372,
           0.008784196339547634, 0.0007335299742408097, -0.02426551841199398, -0.0133942486718297, 0.01721174269914627,
           0.01725137047469616, -0.0006588147371076047, -0.017026811838150024, -0.005795588251203299,
           -0.020659374073147774, -0.0024239099584519863, -0.019034627825021744, -0.011465688236057758,
           0.0075028929859399796, -0.0031900503672659397, 0.024899564683437347, 0.016987184062600136,
           0.008354893885552883, 0.008044474758207798, -0.01121471170336008, -0.013189504854381084,
           0.007119822781533003, 0.016366345807909966, 0.018453417345881462, -0.010382524691522121, 0.03909958153963089,
           -0.000770681188441813, -0.021861422806978226, 0.019932862371206284, -0.007549125701189041,
           0.039258092641830444, 0.006475868169218302, 0.01686829887330532, 0.011941224336624146,
           -0.0017551882192492485, -0.0030232828576117754, -0.10329686850309372, -0.028162267059087753,
           0.017858998849987984, 0.01590401865541935, -0.012396945618093014, 0.018506255000829697,
           -0.013790528289973736, 0.0027293753810226917, 0.0034971670247614384, 0.015534158796072006,
           0.003376631997525692, -0.029536036774516106, -0.017674067988991737, -0.012330899015069008,
           0.01697397418320179, 0.0008841987582854927, 0.0190082099288702, -0.013962249271571636, -0.010910897515714169,
           0.01600969396531582, -0.006895264144986868, 0.008988941088318825, 0.0062447055242955685,
           -0.01372448168694973, 0.006789589766412973, -0.009893779642879963, -0.02340691164135933, 0.01821565069258213,
           0.0031372131779789925, 0.001075733918696642, 0.016445601359009743, 0.01130717620253563,
           -0.0016379555454477668, -0.009662616066634655, -0.018585510551929474, 0.020078163594007492,
           -0.022772865369915962, 0.012912108562886715, 0.02488635666668415, -0.03186087682843208,
           0.0022290723863989115, -0.014411366544663906, -0.01569266989827156, -0.014701971784234047,
           -0.002227421384304762, 0.013539551757276058, -0.023314446210861206, 0.007337776478379965,
           0.0073113576509058475, -0.024569332599639893, -0.016115369275212288, -0.014569878578186035,
           -0.026696031913161278, -0.014649134129285812, 0.006994334049522877, 0.007833126001060009,
           0.006522100884467363, 0.018294906243681908, -0.00947108119726181, -0.012694154866039753,
           -0.019589418545365334, 0.0018410487100481987, 0.0056502860970795155, 0.0007620125543326139,
           -0.030539944767951965, -0.001172326970845461, -0.031094735488295555, -0.017132485285401344,
           0.004735540598630905, 0.006644287146627903, 0.011828945018351078, -0.013632016256451607,
           -0.02561286836862564, -0.000537866959348321, -0.02492598444223404, 0.0009651058353483677,
           -0.010534431785345078, -0.01519071590155363, 0.003969400189816952, -0.05701141804456711,
           0.0005097971297800541, -0.021861422806978226, 0.01763444021344185, -0.03431781008839607,
           0.005372890271246433, 0.01849304512143135, 0.011452479287981987, 0.0013308388879522681,
           -0.0032759110908955336, -0.008599266409873962, 0.00286477105692029, 0.005224285647273064,
           0.005604053381830454, -0.006928287446498871, -0.0071660554967820644, 0.02441081963479519,
           -0.0050558666698634624, -0.018176022917032242, -0.019893232733011246, 0.00324453879147768,
           0.007701032795011997, -0.005904565565288067, -0.046206194907426834, 0.011888386681675911,
           0.01725137047469616, -0.010930711403489113, -0.025388309732079506, -0.012396945618093014,
           0.005917774513363838, 0.012496015056967735, 0.02080467715859413, 0.02458254247903824, -0.017092857509851456,
           0.015256762504577637, 0.017198532819747925, 0.007483079098165035, -0.027264032512903214,
           -0.016551276668906212, -0.00918047595769167, 0.0031471201218664646, 0.012865875847637653,
           0.02018383890390396, -0.012885689735412598, -0.0004326051857788116, -0.018928952515125275,
           0.0030265850946307182, -0.01501899491995573, 0.0020507466979324818, -0.021082071587443352,
           0.032098643481731415, -0.014371738769114017, -0.010072105564177036, 0.03521604463458061,
           -0.012093131430447102, -0.018572302535176277, -0.008341684937477112, -0.0011343501973897219,
           -0.021451933309435844, 0.006406519562005997, -0.01793825440108776, 0.03384227305650711,
           -0.004207167774438858, 0.006617868784815073, -0.040394097566604614, 0.015111460350453854,
           0.011379827745258808, -0.004735540598630905, -0.004712424241006374, 0.0015141181647777557,
           -0.001814630115404725, 0.004124609753489494, 0.005369587801396847, -0.0015801647678017616,
           0.007443450856953859, 0.003181794658303261, -0.01984039694070816, 0.007740660570561886, 0.01496615819633007,
           0.007826521061360836, -0.01338103972375393, 0.015573786571621895, -0.02945677936077118, 0.01159778144210577,
           0.0011128850746899843, 0.015032204799354076, -0.005623867269605398, 0.015705879777669907,
           6.25894681434147e-05, -0.03291762247681618, -0.0075028929859399796, 0.028611384332180023,
           -0.004322749562561512, -0.019100675359368324, -0.016643742099404335, 0.00954373273998499,
           -0.005663495510816574, 0.014701971784234047, 0.0013390947133302689, 0.006449449807405472,
           0.009609779343008995, -0.029668129980564117, 0.021544398739933968, 0.013348015956580639,
           0.004375586751848459, 0.0071660554967820644, -0.015586995519697666, 0.01928560435771942,
           0.004385493695735931, -0.020315932109951973, 0.00035892194136977196, -0.004375586751848459,
           0.00764819560572505, -0.02073862962424755, 0.011729874648153782, -0.005019540898501873, -0.01237713173031807,
           -0.022772865369915962, 0.02668282389640808, -0.025216588750481606, 0.0034839578438550234,
           0.020870722830295563, -0.00031784921884536743, -0.0015950251836329699, -0.006360286846756935,
           -0.010897687636315823, -0.017898626625537872, -0.01187517773360014, 0.005006331484764814,
           -0.012040293775498867, -0.02522979862987995, -0.018915744498372078, -0.0033650738187134266,
           0.014252854511141777, -0.0007904951344244182, -0.0031883991323411465, 0.01701360195875168,
           -0.009702244773507118, 0.002305026166141033, -0.013420667499303818, -0.009986245073378086,
           -0.03656339272856712, 0.012529038824141026, 0.001564478618092835, 0.024873146787285805, 0.017687277868390083,
           -0.016498439013957977, 0.030460689216852188, -0.018995000049471855, 0.015705879777669907,
           -0.0063536823727190495, 0.015494531020522118, -0.03730311617255211, 0.005874844267964363,
           0.027871662750840187, -0.017515556886792183, -0.004078377038240433, -0.01825527846813202,
           0.019444117322564125, -0.004022237379103899, -0.010963734239339828, -0.009768291376531124,
           0.03447632119059563, 0.005092192441225052, -0.015547367744147778, 0.013281969353556633, -0.01221861969679594,
           0.013658435083925724, 0.026603566482663155, -0.001470362301915884, -0.013975459150969982,
           -0.021993516013026237, -0.008942708373069763, 0.002841654699295759, 0.020025325939059258,
           -0.007192473858594894, -0.01372448168694973, 0.015098251402378082, -0.00404535373672843,
           0.002308328403159976, -0.02603556588292122, -0.0011929665924981236, 0.026418637484312057,
           -0.0003665585827548057, 0.02958887256681919, 0.010534431785345078, -0.028188684955239296,
           4.211760369798867e-06, 0.016392763704061508, -0.014147180132567883, -0.013935831375420094,
           -0.01994607038795948, 0.007899172604084015, 0.0014307342935353518, -0.015137879177927971,
           -0.021927468478679657, 0.02956245467066765, -0.006707031279802322, -0.006796194240450859,
           -0.0021531188394874334, 0.022138817235827446, 0.016960764303803444, -0.01014475617557764,
           0.02969454787671566, -0.020870722830295563, -0.034449901431798935, -0.011987456120550632,
           -0.016128577291965485, 0.002179537434130907, -0.0019714906811714172, 0.0016891416162252426]


def from_custom_dataset():
    args = parse_args()
    Dataset.IMAGE_DIR = "images"
    Dataset.ANNO_DIR = "Annotations"
    Dataset.CAPTION_DIR = "texts"
    # generate dataset
    generate_dataset(args)


class Caption:
    def __init__(self, items):
        self.idx = items[0]
        self.file_id = items[1]
        self.filename = self.file_id.replace(".jpg", ".txt")
        self.caption = items[2]
        self.author = items[3]
        self.is_occluded = items[4]
        self.is_error = items[5]


def check_grammar(text):
    # Create a LanguageTool instance
    try:
        from language_tool_python import LanguageTool
        tool = LanguageTool('en-US')

        # Check grammar in the text
        matches = tool.check(text)
        return matches
    except:
        return []


class SQLiteDataWrap:

    def __init__(self, dbpath):
        assert os.path.isfile(dbpath), dbpath
        conn = sqlite3.connect(dbpath)
        self.conn = conn
        self.dbpath = dbpath
        self.is_clean = False
        self.image_path_dict = None

    def clean(self):
        self.is_clean = True
        return self

    def get_path(self, file_id, image_paths):
        if self.image_path_dict is None:
            self.image_path_dict = defaultdict(lambda: None)
        filepath = self.image_path_dict[file_id]
        if filepath is None:
            filtered = list(filter(lambda fname: file_id in fname, image_paths))
            if len(filtered) > 0:
                filepath = filtered[0]
                self.image_path_dict[file_id] = filepath
        return filepath

    def export(self, data_root, clean=False, copy_images=False, image_paths=()):
        data_root = pathlib.Path(data_root)

        # defining paths
        caption_root = data_root / "train" / "texts"
        image_root = data_root / "train" / "JPEGImages"
        if clean:
            print("Cleaning previous data...", end="")
            # deleting directories
            shutil.rmtree(caption_root, ignore_errors=True)
            shutil.rmtree(image_root, ignore_errors=True)
            print("Done")
        # create directories
        os.makedirs(caption_root, exist_ok=True)
        os.makedirs(image_root, exist_ok=True)
        os.makedirs(data_root / "test" / "JPEGImages", exist_ok=True)
        os.makedirs(data_root / "test" / "JPEGImages", exist_ok=True)

        statistics = defaultdict(lambda: defaultdict(lambda: 0))
        # taking DB cursor and running queries
        print("Quering dataset form SQLITE...")
        curr = self.conn.cursor()
        count = curr.execute("SELECT count(*) FROM caption").fetchone()[0]
        dataset = curr.execute("SELECT * FROM caption")
        print("Creating dataset form SQLITE...", )
        for idx, data in enumerate(dataset):
            print("\r{}/{} {}% ".format(idx, (count - 1), round((idx / (count - 1) * 100.), 2)), end="\b")
            caption = Caption(data)
            if copy_images:
                # copying images
                filepath = self.get_path(caption.file_id, image_paths)
                if not os.path.isfile(image_root / caption.file_id):
                    shutil.copyfile(filepath, image_root / caption.file_id)

            # creating the caption files
            with open(caption_root / caption.filename, "a+") as fp:
                fp.seek(0)
                lines = fp.readlines()
                if caption.caption + "\n" in lines:
                    statistics[caption.filename]['duplicate'] += 1
                    statistics["caption"]['duplicate'] += 1
                elif len(caption.caption.split(" ")) < 2:
                    statistics[caption.filename]['faulty'] += 1
                    statistics["caption"]['faulty'] += 1
                else:
                    fp.write(caption.caption + "\n")
                    # check grammar
                    errors = check_grammar(caption.caption)
                    # save statistics
                    statistics[caption.filename]['caption'] += 1
                    statistics[caption.filename]['grammar'] += len(errors) > 0
                    statistics["caption"]['total'] += 1
                    statistics['grammar']['error-sentence'] += len(errors) > 0
                    statistics['grammar']['total-error'] += len(errors)
        print("Done")
        return statistics


def from_sqlite():
    args = parse_args()
    dataset = pathlib.Path(args.dataroot)
    # For reading images
    Dataset.IMAGE_DIR = "JPEGImages"
    # reading filepaths
    file_paths = list(list_dir(str(dataset / "train"), dir_flag=VOC_IMAGES, fullpath=True))
    file_paths.extend(list(list_dir(str(dataset / "test"), dir_flag=VOC_IMAGES, fullpath=True)))
    # generating dataset form SQLIte
    sqlite_data = SQLiteDataWrap(args.sqlite)
    sqlite_data.clean().export(args.data_dir, clean=args.clean, copy_images=args.copy_images, image_paths=file_paths)
    # For generating dataset
    Dataset.IMAGE_DIR = "JPEGImages"
    Dataset.CAPTION_DIR = "texts"
    # generate dataset
    generate_dataset(args)


if __name__ == '__main__':
    # from_custom_dataset()
    # from_sqlite()
    generate_dataset_with_langchain()

""" UBUNTU
sqlite3 -header -csv "C:\\Users\\dndlssardar\\Downloads\\tip_gai_22052023_1743.db" "SELECT * FROM caption" > caption.csv
python data/generate_custom_dataset.py --data_dir data/sixray_sample --emb_dim 300 --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin
python data/generate_custom_dataset.py --data_dir data/sixray_500 --fasttext_model /data/fasttext/cc.en.300.bin  --sqlite /data/sixray_caption_db/<tip_gai.db> --clean --copy_images --dataroot /data/Sixray_easy/
python data/generate_custom_dataset.py --data_dir data/sixray_500 --fasttext_model /mnt/c/Users/dndlssardar/Downloads/Fasttext/cc.en.300.bin --sqlite data/tip_gai.db --clean --copy_images --dataroot "/mnt/c/Users/dndlssardar/OneDrive - Smiths Group/Documents/Projects/Dataset/Sixray_easy"
"""
