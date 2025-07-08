import sys
import os
import numpy as np
import json
import re
from datetime import datetime
from scipy.stats import skewtest, mannwhitneyu, fisher_exact
from datetime import timedelta
import copy
import spacy
import shutil
import textstat
from textcomplexity import surface
from textcomplexity.utils.text import Text
from collections import namedtuple
import nltk
Token = namedtuple("Token", ["word", "pos"])
nlp = spacy.load("en_core_web_sm")


sys.path.append('../application/study_app')
from study.pages import SLEEPCYCLE_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, BLACKHOLES_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, QUARTO_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE
from study.pages import SLEEPCYCLE_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, BLACKHOLES_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE, QUARTO_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE
from study.pages import COCONSTRUCT_POST_QUESTIONNAIRE_PAGE

import sqlite3
import csv
from operator import itemgetter

os.makedirs("user_study_data", exist_ok=True)

correct_answers = {"sleep": {"post_obj_comprehension": SLEEPCYCLE_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE,
                             "post_enabledness": SLEEPCYCLE_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE,},
                    "blackholes": {"post_obj_comprehension": BLACKHOLES_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE,
                             "post_enabledness": BLACKHOLES_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE,},
                    "quarto": {"post_obj_comprehension": QUARTO_OBJECTIVE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE,
                             "post_enabledness": QUARTO_OBJECTIVE_CHOICE_UNDERSTANDING_POST_QUESTIONNAIRE_PAGE,}}

explananda = {"sleep": "The human sleep cycle and its stages",
              "blackholes": "The formation of black holes ",
              "quarto": "The board game Quarto and its rules"}


understanding_questionnaires = {}

for topic in correct_answers:
    understanding_questionnaires[topic] = {}
    for questionnaire in correct_answers[topic]:
        understanding_questionnaires[topic][questionnaire] = {}
        for question in correct_answers[topic][questionnaire]["questions"]:
            understanding_questionnaires[topic][questionnaire][question["question_id"]] = {"text": question["prompt"], "choices": question["options"], "correct_answer": question["correct_answers"][0]}
understanding_questionnaires

with open("user_study_data/understanding_questionnaires.json", "w") as file:
    json.dump(understanding_questionnaires, file, indent=4)

db_name = '../application/study_app/db_finalstudy.sqlite3'

conn = sqlite3.connect(db_name)
cursor = conn.cursor()

cursor.execute(f"SELECT user_id, prolific_pid, study_topic, system_prompt, form_id, question_id, answer FROM study_userresponse JOIN study_usersession ON study_userresponse.user_session_id = study_usersession.id ORDER BY study_usersession.created_at")
rows = cursor.fetchall()

columns = [description[0] for description in cursor.description]

questionnaires_results = {}
setup = {}

accepted_user_ids = ["009d4e0c-ed33-44be-95bf-4623d32cce7e", "00e3f63b-b7e6-4144-87a2-a97f23ba1674", "018e021a-8bd9-4d9a-9a4a-308d89d9375c", "02988131-47f9-402b-ab26-70a504724791", "030d5358-8844-4e97-ac0d-51640fa9fe17", "035e91a0-0889-47a1-a9b7-c733eb9dcfa5", "03e0e065-3b73-48e5-bbed-269b48ddb31e", "072d12e9-f099-43da-8a14-5cdb106196aa", "0cf8f229-08be-43e6-b873-daafa656ff47", "0d708720-2feb-4a21-b52c-66abd80be993", "103f02d7-27f0-4f5b-9a17-36a87e29e082", "10e17f29-174c-48d4-a13f-51fd289001ad", "1129fd80-525f-42b0-8b75-13808f2cee6a", "11346d47-eeeb-419d-9c40-d544f3a46e11", "125122db-3de3-4500-a327-097d48bbbd32", "12907f52-c62c-4c45-980b-edfe73ffa8eb", "13132269-f919-4921-b08b-90651d7d7070", "1319368f-31b1-4fd0-8f88-8faa7d9aeabc", "13d85ac2-4e4b-4a1e-a5d0-075594c9a530", "144fa100-fad9-4484-a08f-dce9d1b9f3a7", "14c9d525-4b96-4f11-b7a9-8d1828eeabf5", "157030ab-d5cc-4ee1-a3a4-e61c8ac51b72", "15a0a015-4f0e-425c-a7c4-93e5f0a7b002", "16660a11-9732-4b4b-a63f-cc6fc6a37ccc", "17a19874-a256-418c-b8af-4eacce4cbf40", "1a2f56b6-9088-46be-8a70-6fae77b127fe", "1ac58ead-6e46-420d-bf38-0219ecc99c5f", "1b81e099-13f3-47bf-ad36-0cc8c9e9adec", "1b8d9f81-7e84-4db1-b478-3528f95e398a", "1ce164f4-d670-4cf0-91fb-8eebed90bef8", "1dfc586a-890c-4145-ab25-1dfab9f12c4d", "1e2912e8-2060-46ad-bdcb-4234cf3440ca", "1ea378da-2ccd-4c37-ad34-241ab7c3edc1", "232e77ec-25c0-41e9-a7e5-99804c0bdcdf", "2375381f-5a90-4164-904e-1cd0a47cbb7d", "24174b59-7360-4c00-b6b7-33f168fd7259", "25793c10-5312-49a0-bce8-c5f9304e87b5", "25d87fb5-f4aa-4873-b420-05ec53aab902", "260ea698-1b2b-423e-b5b8-183bce78e049", "28983176-7112-4bfe-9331-f6f58a3a89cb", "2900b6fc-6f77-42aa-a728-eb12f32903de", "29f8d5d4-5cfa-4486-a1db-37d72494e63c", "2bcddb72-46d6-4478-8dbb-3335a67b2aa0", "2c9e5aaf-fb26-4e0b-8c46-002d916d0128", "2d7713bd-88ec-4cf0-ad3c-8a095b950a74", "2ec19a19-df8b-4c49-b91b-ea326f29ceb4", "2efeaaf0-4441-4319-bcee-c75c6a0a5565", "2fc51e13-6da1-4f09-b302-fe739b2b985e", "30a8c97f-844e-498a-b6d0-ea4b94155e2c", "31479367-972d-4be0-a5b7-b545b8667bd1", "32121363-8a0d-48ab-b558-9036bd13e58c", "32ce18f2-c2bb-4894-b29b-3e62d1a9ee3a", "3740acf8-0f72-4107-b17a-26fd0d1926da", "37785049-dd47-49bb-a756-2fcf89383a82", "38ce7ff6-90ec-432b-909c-c3bd67fb843d", "3a0e392e-2dbb-4492-a587-9ec627566b3c", "3b57a2ab-8ce3-42dd-a777-b2841861bde1", "3c178a33-06f1-47b4-a1fb-1de4a61909d8", "3ccf315e-b8cd-4f4a-9ad5-12d782d72013", "3ceae169-11e4-4ef3-8c2c-aa79584a5936", "3d7f6dba-32ed-4ec5-ab2a-80ecbf1e2de3", "3dc8c94d-148d-491f-a1ca-248650bc6888", "3e7be408-e3f9-405f-a991-0d87f37cd189", "3ec11494-f5eb-4415-94c5-20ca29c05af1", "3eed5e21-d83a-4a90-8be2-8ec56c065ca9", "3f4519d8-4477-4a52-9793-2299dcf8d12b", "403d511f-f6a9-4a9f-920d-046394988bc1", "41c9b18f-f0a6-41ba-81b5-b6ca49c65a15", "4258edbc-7e30-4cdf-a527-0d8fb56b5118", "435ba493-2dac-4d10-bc1a-1eed43766dee", "43b62d26-b4b6-4346-aa06-7a9fc77cc7f3", "45ce6553-a658-4701-8a83-d18b9e587b62", "46e15669-d95b-4e6a-a8a7-45f231ed144a", "476115fa-cc5a-46a6-80f4-aa11ab376208", "47c127f9-4312-4bb0-926b-8ce6e9f20eb5", "4804a373-2fef-4e1b-8071-ca82bb09b414", "491a5c27-042c-4f8b-9335-a5ddf94595f7", "49a7e71c-1eb5-495d-8eb8-75f0aa426154", "4a310fa2-f370-4c7a-b56d-c40b15b85789", "4a852b05-dd65-4df6-9dc9-58955c0396ab", "4d14f287-01ff-4bc0-adaa-858fc09c69ee", "5192d75a-6e1c-45df-9f6d-0829da458138", "53de6de0-0280-4964-ab3d-9c864f601c03", "5428d30b-b0d0-4d0f-a3ef-128d47b44440", "5563ff7a-2de3-4e36-9e21-59fecadd020f", "564e0c57-69b5-4d1c-8641-24ad63b7c938", "58c8e952-90cd-493c-bca1-3da736eb284b", "59194fe4-38ab-41ed-9d03-564ae25ad08f", "59f05a5a-ce18-415b-a9e4-8f04b5ae3613", "5acbaa4c-f600-43ff-8961-2f5ff48bca0f", "5b4f2db1-5ba9-4662-a052-ba0f16055523", "5c188ac7-063d-49a2-8cde-7438272af94d", "5cb5e400-7d54-4b3a-bb5c-913704593786", "5e2e15cd-b199-4d77-a150-5cdb46606993", "5ed16d94-a741-4815-ad29-d5423672b8de", "5f326629-9a39-4ccf-9a0d-01b94cf8f8bb", "6045a1d2-0bb1-4c6f-ac51-520d8964b621", "609b81c7-8edc-4531-88ad-1c580a2ab6dc", "626a2cdd-0ae9-4c93-a2c9-91e778df7446", "63e3557e-7150-420a-bd13-ac53579b76d4", "6495241b-fae9-4500-a954-6e35f115e0fe", "6609c50f-2cb1-4dcf-9bee-8ab09fde629e", "67cec50b-5672-4563-ab75-9cd876ac48a2", "69d9d20b-36d9-41b5-9123-ffa4ffd00435", "6af8bafa-716f-4cf4-a367-f188b8513e8e", "6e21b32d-8e58-460c-9d15-efadae18090f", "6e49c82d-1f45-47ac-8908-59a8605c3a80", "6fd21fd7-02fe-406b-bf4b-e051d24225df", "703bc254-fff5-4596-846e-db493114c4f0", "71ca5537-bd2e-41f6-b85d-d45a74f305ef", "72ba75af-0cbc-43af-bee0-c472aad7d1d4", "72d0960b-9c87-4c2b-b70f-b70d292ef180", "73674d0f-a436-4b1e-852f-fec1461e8fa4", "738a27a1-18bb-4b58-a4e8-6ee15aa239e2", "74bbcd8a-2fcb-4c84-9202-fb0e9984e66a", "74ce501b-8069-496b-bb71-d09ec894ee74", "7569a07b-5557-4ff4-b884-58293aa74f9a", "7631340b-f87e-412b-96c7-e6fe067e0e62", "7a514317-788b-4969-bb1d-4ae222875870", "7c0715cc-9954-45e6-968b-e089b2f8c9c3", "7c29780d-39b3-404e-a515-c3ff939f7172", "7c4e31c4-0158-4bf1-bf38-5547a05aa165", "7e2841b4-1b54-4ed8-813d-6b1e8af3449c", "804d5002-c05b-4cb5-bd0d-ffb1d54cfcce", "80880e49-1023-4843-be9e-514d9bacc03a", "80972446-064a-405d-ae9e-e26f2d19110e", "8244adeb-25c7-4f0c-abdc-ade9b9cb6957", "82c46020-8c3e-4cfe-86fc-688fc658339d", "83f34f21-6490-444b-aa43-823f2e59078b", "85747006-3169-434e-99f2-22d9424de5dd", "85c3efac-9e8a-4ce8-aab8-a595cd8367d9", "85f904ac-87ae-4beb-943c-cd40c1763b9e", "8659cd90-4161-43cf-b1c6-090d91789b1d", "87633b78-e8ae-4355-9f89-96d990239549", "879632db-0b30-414b-809b-3cfe2e9818f3", "8a6348ec-e1d7-4945-80d1-0e631d8273d9", "8a95072c-3466-481a-9dce-03f18d83dcdb", "8c2ab69d-e51b-47d2-a605-8d7612b8ae33", "8d9f132f-1d0e-4985-9cb7-7076284e3326", "8ddc37e6-01a7-4dc5-b851-98fdc06fbf71", "8def52dc-7f21-466f-89cd-403fc571dd70", "8e61a74e-15b5-43e5-aaef-e7671585481f", "905232e8-c5b1-4a43-bb07-f8f1a2939c5c", "9197e989-36ac-40f2-ab5a-d55712c89f6d", "91b2beb3-3768-49c9-9a37-7e19b50288a1", "91b8b8ad-433e-4d9d-8912-c84c3a2e9f30", "91f75920-b7f7-4824-b5e5-9932b1b27b21", "936e6d84-c40a-451d-948a-0c1d72a4d371", "943a0efe-d0e8-45ba-9022-cda101802d54", "94e00fdc-6389-463a-bdee-0ea391ad89a4", "9573e665-18b3-4a34-972b-d037f51fd315", "95d9089d-0496-407c-9d04-04bbf6cbe8fc", "989344cb-41fd-4a9d-af0b-8d683eeddff7", "996f2be4-5deb-42e9-b82e-dc777b0516b8", "99f1aead-17bc-4aed-82c4-223794cc2bf8", "9bb0665e-8f40-4164-a750-9264d434aad6", "9c9fefc9-3ef9-4b39-8ec7-1ca3fd4b0145", "9e082f9a-76c8-4b9d-bc27-021f9ac2d4aa", "9e77286e-1614-435d-9819-02582fed5748", "9ed98cfb-ec03-49b2-8a17-6dcb06ab696a", "a1e63abe-6f61-40ea-8d28-6503b7a66a24", "a212bef1-65a3-4880-b2c5-14dbbdbca1b2", "a3200333-5f19-4933-8e74-8640248338d2", "a499f6f9-942f-431e-ba41-b85b70158b4f", "a695a53c-5f3c-40f8-ba93-c50d96a0fd1a", "a8fc7df9-983e-4149-9d39-532b9ca69f13", "a95310f1-aaf4-4f32-9f1a-f22e6bb48097", "aa7cce6d-9e66-4ec7-a38c-05faa89f7b51", "aac9684d-7b90-419d-b76c-24772bafe767", "ad71fb47-130b-492d-8e4d-3a36dd797e00", "adc767e6-0c90-49f5-a858-4cfcc96649bd", "adf7bc16-53ec-4762-a496-e6430fdc6a4a", "aebfb9d6-057e-4a0d-b613-e5fa08feaf2d", "af3763c7-5d9e-4f93-9c1c-077196f015f4", "af810b0d-f8d9-4543-a777-310745978bc2", "b1a9f72c-2895-41c4-934c-818efec77b0a", "b1fe423c-79b9-46ce-87ee-8985a6edb566", "b374b420-7c85-4a2e-b9a4-2f2908e152fd", "b3c3df7e-272d-4d34-8eee-a42ea417f927", "b4e4fb1d-56d1-46cd-a7b4-f7a82671c845", "b5a2e301-30fa-44c8-b2ba-cadb0016d712", "b824accb-dea5-4dfe-ad58-1e0030838e27", "b933306b-ecc7-40b1-a263-f0a0cf9aaa88", "b959304a-de99-459d-8e22-07c21780e181", "ba5bb21e-fb2d-4893-8aad-8f27b33c161d", "baa1511d-168d-4a19-a911-2891ab444f97", "bb896dfe-3529-42f1-9eef-39c1a7b5cd81", "bc5fdd28-517e-4e5a-8eff-70e811435a23", "bc8e7786-6cb7-4411-8ca4-28362cd604e4", "bcfbb88e-f0aa-40b4-b7dc-073c123f6880", "be9189fb-ef3a-462e-801b-886d368b1b44", "bf93610a-ca9f-4f40-8d35-534cedb5fca2", "bfb677e5-c44a-4fd0-aa35-6ebf9a3b796d", "c4add14c-3bce-4cbc-ae74-994438d14fbe", "c4d65a9f-89e9-4614-9bfa-c4b8b43b49a5", "c788da23-8b1b-45aa-a932-b8af087c17a6", "c7ba2145-fdb2-4c36-ab5c-a858a103b0a9", "c7fb32c3-19c9-418f-8292-e924fd0b4b65", "cac86520-d0c8-4ca2-8f04-29d780416bbb", "cba03c12-17f0-4de9-8925-30e62899e746", "cc35ad1d-b375-4dbf-a628-8f9f4b78354f", "cdc6b64a-251d-4942-9c20-6509823305db", "cdcec3ad-6652-41b5-8796-cd79a6199cb4", "cf39e702-e86e-4733-824b-c6f8c2b5043e", "d00c6fa1-cc16-4c6d-b1d5-69acbb23c838", "d15b5de8-1130-4485-a470-a14ce3339ea1", "d1cfd841-00c8-4908-8a1e-d9c6005245d7", "d2e9e184-14ab-4ba8-b330-86869b5d2feb", "d2f10222-f397-4aae-ba21-81e79e377291", "d4571b61-ffb7-4320-987b-057dcf6d50db", "d73d27b2-d567-400a-b841-63e5dc797aeb", "d77695f4-80e2-40a7-bf7b-9ff97e0f5b92", "d8e89cbb-2560-42d2-9f57-6010845313e8", "d8f33103-85bb-49be-86ae-50b589fcc92f", "d96f67fe-50e2-40b0-bbe0-b985abe4586c", "da64b293-2c59-4dcb-8758-139e6bec6c18", "db020bea-c4a2-493f-b80c-58008eac18e4", "dbf907e0-7c2b-4649-bfe3-a6761cf66b9d", "dcd4fb27-0490-4321-b513-12d25f3e6249", "dde79fab-585c-4aeb-8d45-7ff328892bfb", "decb820f-9e34-48b6-b05a-d3de66ba0c75", "e125fdba-64c4-450d-a565-7e8d2dab62c3", "e1bb048f-13e1-42dd-a755-910b2c154de1", "e23a2cb9-cff4-4843-876f-575da8477039", "e2abf276-471a-412f-b496-08ebf973a299", "e2c8a3e4-03f1-4fa6-9e50-c3d4434a7475", "e48542d5-0782-4ede-95d8-8cf437ea5a47", "e5e34341-6055-44d4-a570-96f38e8cc1b6", "e6321fb0-1435-4d16-b071-0239078f806e", "e757bc62-b3ee-4c7f-a8d5-e05c641dd7b1", "e8009613-e10b-41c8-bd2a-6498f4a79e9b", "e81033b0-bb31-47b4-b68d-198c9851ca3f", "e8540d5a-54dd-44a7-931b-daa7ff94b97c", "eb6f4cb9-dbd8-4190-b131-b7adc66b38e5", "ec1c12a6-fa71-44d6-a828-19ae8b5c5b82", "ed35e203-2137-43e0-b5c0-44a9859dc9c2", "ee2fc6bc-c093-4238-8980-04fab17bd729", "efd3159c-1330-4923-96a6-9ed5e50ef4e3", "f039f2a0-5318-45a6-a0cd-f92af2fb9b12", "f03b876b-5b93-424f-840d-6abed8c13886", "f1ab80ea-ad6a-4749-89da-4da7dbfc5cbe", "f2663a1d-94ad-4dc7-96ea-12e7376110d5", "f2926fa2-6d05-4e17-a9ea-1f8baa003e85", "f2a72ee9-33f5-4598-9677-3eb92f2be2f2", "f36d588a-8a22-4f55-b777-c491746245a8", "f42992ff-cafc-46c4-a36f-17870e2ee91b", "f543eab6-3867-44a2-a841-8fdf15df65ad", "f675676a-e42e-43af-930c-a4458505a96f", "f8860777-e2bb-4599-abf3-1fd80edca92f", "f89861d4-4104-4586-97dd-150e85126000", "fb6f5475-e27a-4007-877e-314c296cf27d", "fc92da47-1758-413d-b0f0-ebdfcecf77e0", "fc96fab7-b61e-4eaa-bb79-d2e6a148af95", "ff22c1f9-763d-4429-8733-5d3121fd9d8b", "ff3ddcae-fee7-45e9-90b6-387e65f12f89", "ff7ff44a-218c-45e6-8364-c9ddb79abeda", "0ae43654-ee44-470f-baf6-4fc7ec2931bf", "0dcfd949-8b0d-45d8-b7e8-ae635ea02a31", "17c0d7c1-d6b7-4d38-b3f9-e67c0b7acf13", "1d757a56-a276-4b83-8d31-c3af0c5cc59c", "1f0180c3-61fd-4c09-9c97-a8d281dd061f", "2d663df3-3180-4b5f-98bc-89d4f3c28214", "30dccb8a-6433-4444-96aa-db5538801f2c", "39f7e450-ae9f-4f16-897d-53050bac1f65", "3bd78fb5-fb41-4a94-bf8e-8a3582fe590c", "4be69e0e-7764-4a53-a350-dfcf9796e35e", "4feb26df-6c7f-464e-9cfe-2b47531fb4d7", "55a64984-b356-4ee8-b0c3-ceeabcd0e73e", "6509602a-da51-44f9-b97d-ac1386802527", "97607f82-6ea1-45d9-9bf2-ba58244340d2", "99638a8d-420b-419f-9cb3-c3bafea036ec", "acfa4cd6-137b-4505-b71e-1fa6084487fe", "adc2cacf-b495-4f4f-8945-9a11ca8fe2a9", "ae61b41f-863d-4dfa-97c5-4b2d586af6f8", "ba9a634d-6644-4808-8bf1-ab97375c7b54", "e4f11cc9-d487-4055-9b1c-40ea7978114d", "f9006a64-d676-4aff-938d-c3ddf0dcdc02"]

for row in rows:
    result_dict = dict(zip(columns, row))
    user_id = result_dict["user_id"]
    if user_id not in accepted_user_ids:
        continue
    #prolific_id = result_dict["prolific_pid"]
    topic = result_dict["study_topic"]
    prompt = result_dict["system_prompt"]

    if user_id not in questionnaires_results:
        questionnaires_results[user_id] = {"pre_subj_comprehension": {},
                                        "pre_motivation": {},
                                        "post_subj_comprehension": {},
                                        "post_motivation": {},
                                        "post_obj_comprehension": {},
                                        "post_enabledness": {},
                                        "post_constructiveness": {}}
        setup[user_id] = {"topic": topic, "explanandum": explananda[topic], "setting": prompt}

    if "subjective_understanding_pre" == result_dict["form_id"]:
        if result_dict["question_id"] in ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]:
            questionnaires_results[user_id]["pre_subj_comprehension"][result_dict["question_id"]] = result_dict["answer"]
        else:
            questionnaires_results[user_id]["pre_motivation"][result_dict["question_id"]] = result_dict["answer"]
    elif "subjective_understanding_post" == result_dict["form_id"]:
        if result_dict["question_id"] in ["q1", "q2", "q3", "q4", "q5"]:
            questionnaires_results[user_id]["post_subj_comprehension"][result_dict["question_id"]] = result_dict["answer"]
        else:
            questionnaires_results[user_id]["post_motivation"][result_dict["question_id"]] = result_dict["answer"]
    elif "objective_understanding_post" == result_dict["form_id"]:
        questionnaires_results[user_id]["post_obj_comprehension"][result_dict["question_id"]] = result_dict["answer"]
    elif "objective_understanding_post_choice" == result_dict["form_id"]:
       questionnaires_results[user_id]["post_enabledness"][result_dict["question_id"]] = result_dict["answer"]
    elif "coconstruct_post" == result_dict["form_id"]:
        for question in COCONSTRUCT_POST_QUESTIONNAIRE_PAGE["questions"]:
            question_id = question["question_id"]
            if question_id == result_dict["question_id"]:
                if question_id == "q13":
                    if result_dict["answer"]:
                        questionnaires_results[user_id]["post_constructiveness"][result_dict["question_id"]] = result_dict["answer"]
                    else:
                        questionnaires_results[user_id]["post_constructiveness"][result_dict["question_id"]] = "<no answer>"
                else:
                    questionnaires_results[user_id]["post_constructiveness"][result_dict["question_id"]] = result_dict["answer"]
    elif "external_sources_open_question_1" in result_dict["form_id"]:
        if result_dict["question_id"] == "q1":
            questionnaires_results[user_id]["post_obj_comprehension"]["open-question-q1"] = result_dict["answer"]
        elif result_dict["question_id"] == "q2":
            if result_dict["answer"]:
                    questionnaires_results[user_id]["post_obj_comprehension"]["open-question-q2"] = result_dict["answer"]
            else:
                questionnaires_results[user_id]["post_obj_comprehension"]["open-question-q2"] = "<no answer>"
    elif "external_sources_open_question_2" in result_dict["form_id"]:
        if result_dict["question_id"] == "q1":
            questionnaires_results[user_id]["post_enabledness"]["open-question-q1"] = result_dict["answer"]
        elif result_dict["question_id"] == "q2":
            if result_dict["answer"]:
                    questionnaires_results[user_id]["post_enabledness"]["open-question-q2"] = result_dict["answer"]
            else:
                questionnaires_results[user_id]["post_enabledness"]["open-question-q2"] = "<no answer>"

conn.close()

with open("user_study_data/questionnaires_results_per_user.json", "w") as file:
    json.dump(questionnaires_results, file, indent=4)

with open("user_study_data/setup_per_user.json", "w") as file:
    json.dump(setup, file, indent=4)


conn = sqlite3.connect(db_name)
cursor = conn.cursor()
query = """
SELECT user_id, message, is_user_message, timestamp
FROM (
    SELECT 
        id,
        message,
        is_user_message,
        timestamp,
        user_session_id,
        ROW_NUMBER() OVER (PARTITION BY user_session_id ORDER BY timestamp) AS row_num
    FROM study_chatmessage
)
JOIN study_usersession ON user_session_id = study_usersession.id
ORDER BY user_session_id, row_num
"""
cursor.execute(query)
rows = cursor.fetchall()

chat_messages = {}
for row in rows:
    user_id = row[0]
    if user_id not in accepted_user_ids:
        continue
    if user_id not in chat_messages:
        chat_messages[user_id] = []

    message = row[1]
    is_user_message = row[2]
    timestamp = row[3]
    
    chat_messages[user_id].append((is_user_message, timestamp, message))

conn.close()


all_chats = {}

for user_id in chat_messages:   
    if user_id not in all_chats:
        all_chats[user_id] = []

    start_timestamp = chat_messages[user_id][0][1]
    for index, (is_user_message, timestamp, message) in enumerate(chat_messages[user_id]):
        message = message.replace("\n\n", "\n")
        if is_user_message:
            current_turn = {
                "turn_id": index,
                "timestamp": timestamp,
                "turn_text": {
                    "author": "Explainee",
                    "text": message
                }
            }
        else:
            current_turn = {
                "turn_id": index,
                "timestamp": timestamp,
                "turn_text": {
                    "author": "Explainer",
                    "text": message
                }
            }
        all_chats[user_id].append(current_turn)

with open("user_study_data/chat_per_user.json", "w") as file:
    json.dump(all_chats, file, indent=4)
   