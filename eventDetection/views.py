from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pickle
import re
import os
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier

def index(request):

    with open(os.path.join(settings.TEXT_ROOT,"featureDictionary.txt"), "rb") as myFile:
        feature_dictionary = pickle.load(myFile)


    with open(os.path.join(settings.TEXT_ROOT,"featureLengths.txt"), "rb") as myFile:
        feature_lengths = pickle.load(myFile)

    with open(os.path.join(settings.TEXT_ROOT,"bigrams.txt"), "rb") as myFile:
        bigrams = pickle.load(myFile)

    with open(os.path.join(settings.TEXT_ROOT,"teenVictim.txt"), "rb") as myFile:
        victim1 = pickle.load(myFile)

    with open(os.path.join(settings.TEXT_ROOT,"adultVictim.txt"), "rb") as myFile:
        victim2 = pickle.load(myFile)

    with open(os.path.join(settings.TEXT_ROOT,"wasVictime.txt"), "rb") as myFile:
        wasVictim = pickle.load(myFile)

    dataset = np.zeros((1, len(bigrams) + 3))

    current = 0
    headers = bigrams
    headers.append('victim1')
    headers.append('victim2')
    headers.append('victim3')
    headers.append('label')


    news = 'ভারতের মধ্য প্রদেশের সাতনা জেলায় গোহত্যার অভিযোগে রিয়াজ নামে এক যুবককে গ্রামবাসীরা পিটিয়ে হত্যা করেছে বলে অভিযোগ উঠেছে। গুরুতর আহত অবস্থায় আরো একজন হাসপাতালে ভর্তি করা হয়েছে। এবিপি আনন্দের প্রতিবেদনে বলা হয়- বৃহস্পতিবার সন্ধ্যায় রিয়াজ ও শাকিল নামে দুই যুবক গরু মাংস নিয়ে যাচ্ছিলেন। তারা আমগড় গ্রামের পৌঁছালে গ্রামের কয়েকজন যুবক তাদের ধরে প্রচণ্ড মারধর করে।  শুক্রবার ভোরে রিয়াজ মারা যান। শাকিলের অবস্থা আশঙ্কাজনক। তাকে হাসপাতালে ভর্তি করা হয়েছে। সাতনার পুলিশ সুপার জানিয়েছেন, এ ঘটনায় ৪-৫ জনকে আটক করে জিজ্ঞাসাবাদ করা হচ্ছে। ঘটনাস্থল থেকে জবাই করা একটি গরুর দেহ ও এক বস্তা গরুর মাংস উদ্ধার করা হয়েছে। এর আগে জানুয়ারিতেও বিহারের মুজফফরপুরে উত্তেজিত জনতা একটি ট্রাকে ভাঙচুর চালায়, মারধর করে চালককে। তাদের সন্দেহ ছিল, ওই ট্রাকে গোমাংস নিয়ে যাওয়া হচ্ছে'
    news1 = 'ফতুল্লায় কিশোরী ধর্ষণের শিকার হয়েছে। মঙ্গলবার রাতের এ ঘটনায় বুধবার অভিযান চালিয়ে পুলিশ স্বাধীন (২৫) ও তানভীর নামে দুজনকে গ্রেফতার করেছে। তানভীরকে থানায় নেওয়ার পথে সাংবাদিকদের জানান, সম্রাট নামে এক যুবক তার বান্ধবীকে নিয়ে ফতুল্লার নন্দলালপুর এলাকায় ঘুরছিল। তাদের পথরোধ করে পরিত্যক্ত বাড়িতে নিয়ে যাওয়া হয়। সেখানে সম্রাটকে বাইরে রেখে স্বাধীন ওই কিশোরীকে ধর্ষণ করে। এরপর আমি ধর্ষণ করতে ওই কক্ষে ঢুকলে ওই কিশোরী আমার পায়ে ধরে ভাই বলে ডাকে। এতে ওই কিশোরীকে গালাগালি করে বের হয়ে আসি। একই কারণে আমাদের বন্ধু শান্তও ধর্ষণ করেনি।ফতুল্লা মডেল থানার পরিদর্শক (তদন্ত) শাহ জালাল বলেন, আটকরা জিজ্ঞাসাবাদে ধর্ষণের কথা স্বীকার করেছে। এ ঘটনায় আরো যারা জড়িত, তাদেরও গ্রেফতারের চেষ্টা চলছে'
    news = news1
    news = re.sub('[()|//|\u200c|\u200b|০-৯|\xa0|।|,|-|\r|\n|adsbygoogle|‘|window.adsbygoogle]', '', news)

    words = news.split(" ")

    for i in range(0, len(words)):

        if words[i] in feature_dictionary:

            lenghts = feature_lengths.get(words[i])

            for lens in lenghts:

                bgrm = words[i]

                for j in range(i + 1, i + lens):

                    if j < len(words):

                        bgrm = bgrm + ' ' + words[j]
                    else:

                        break

                if bgrm in feature_dictionary.get(words[i]):

                    if dataset[current][bigrams.index(bgrm)] == 0:
                        dataset[current][bigrams.index(bgrm)] = 1
    for i in range(0, len(words)):

        if words[i] in victim1:

            if dataset[current][1088] == 0:
                dataset[current][1088] = 1
        elif words[i] in victim2:

            if dataset[current][1089] == 0:
                dataset[current][1089] = 1

        elif i + 1 < len(words):

            vbgrm = words[i] + ' ' + words[i + 1]

            if vbgrm in wasVictim:

                if dataset[current][1090] == 0:
                    dataset[current][1090] = 1


    classifier = pickle.load(open(os.path.join(settings.TEXT_ROOT,"classifier.txt"), 'rb'))
    label = classifier.predict(dataset)
    label = str(label)
    f = 'gooooooo '+ label

    return render(request,'text/index.html')


def event(request):
    if(request.method=='POST'):
        news = request.POST.get('news')

        with open(os.path.join(settings.TEXT_ROOT,"featureDictionary.txt"), "rb") as myFile:
            feature_dictionary = pickle.load(myFile)


        with open(os.path.join(settings.TEXT_ROOT,"featureLengths.txt"), "rb") as myFile:
            feature_lengths = pickle.load(myFile)

        with open(os.path.join(settings.TEXT_ROOT,"bigrams.txt"), "rb") as myFile:
            bigrams = pickle.load(myFile)

        with open(os.path.join(settings.TEXT_ROOT,"teenVictim.txt"), "rb") as myFile:
            victim1 = pickle.load(myFile)

        with open(os.path.join(settings.TEXT_ROOT,"adultVictim.txt"), "rb") as myFile:
            victim2 = pickle.load(myFile)

        with open(os.path.join(settings.TEXT_ROOT,"wasVictime.txt"), "rb") as myFile:
            wasVictim = pickle.load(myFile)

        dataset = np.zeros((1, len(bigrams) + 3))

        current = 0
        headers = bigrams
        headers.append('victim1')
        headers.append('victim2')
        headers.append('victim3')
        headers.append('label')


        # news = 'ভারতের মধ্য প্রদেশের সাতনা জেলায় গোহত্যার অভিযোগে রিয়াজ নামে এক যুবককে গ্রামবাসীরা পিটিয়ে হত্যা করেছে বলে অভিযোগ উঠেছে। গুরুতর আহত অবস্থায় আরো একজন হাসপাতালে ভর্তি করা হয়েছে। এবিপি আনন্দের প্রতিবেদনে বলা হয়- বৃহস্পতিবার সন্ধ্যায় রিয়াজ ও শাকিল নামে দুই যুবক গরু মাংস নিয়ে যাচ্ছিলেন। তারা আমগড় গ্রামের পৌঁছালে গ্রামের কয়েকজন যুবক তাদের ধরে প্রচণ্ড মারধর করে।  শুক্রবার ভোরে রিয়াজ মারা যান। শাকিলের অবস্থা আশঙ্কাজনক। তাকে হাসপাতালে ভর্তি করা হয়েছে। সাতনার পুলিশ সুপার জানিয়েছেন, এ ঘটনায় ৪-৫ জনকে আটক করে জিজ্ঞাসাবাদ করা হচ্ছে। ঘটনাস্থল থেকে জবাই করা একটি গরুর দেহ ও এক বস্তা গরুর মাংস উদ্ধার করা হয়েছে। এর আগে জানুয়ারিতেও বিহারের মুজফফরপুরে উত্তেজিত জনতা একটি ট্রাকে ভাঙচুর চালায়, মারধর করে চালককে। তাদের সন্দেহ ছিল, ওই ট্রাকে গোমাংস নিয়ে যাওয়া হচ্ছে'
        # news1 = 'ফতুল্লায় কিশোরী ধর্ষণের শিকার হয়েছে। মঙ্গলবার রাতের এ ঘটনায় বুধবার অভিযান চালিয়ে পুলিশ স্বাধীন (২৫) ও তানভীর নামে দুজনকে গ্রেফতার করেছে। তানভীরকে থানায় নেওয়ার পথে সাংবাদিকদের জানান, সম্রাট নামে এক যুবক তার বান্ধবীকে নিয়ে ফতুল্লার নন্দলালপুর এলাকায় ঘুরছিল। তাদের পথরোধ করে পরিত্যক্ত বাড়িতে নিয়ে যাওয়া হয়। সেখানে সম্রাটকে বাইরে রেখে স্বাধীন ওই কিশোরীকে ধর্ষণ করে। এরপর আমি ধর্ষণ করতে ওই কক্ষে ঢুকলে ওই কিশোরী আমার পায়ে ধরে ভাই বলে ডাকে। এতে ওই কিশোরীকে গালাগালি করে বের হয়ে আসি। একই কারণে আমাদের বন্ধু শান্তও ধর্ষণ করেনি।ফতুল্লা মডেল থানার পরিদর্শক (তদন্ত) শাহ জালাল বলেন, আটকরা জিজ্ঞাসাবাদে ধর্ষণের কথা স্বীকার করেছে। এ ঘটনায় আরো যারা জড়িত, তাদেরও গ্রেফতারের চেষ্টা চলছে'
        # news = news1
        news = re.sub('[()|//|\u200c|\u200b|০-৯|\xa0|।|,|-|\r|\n|adsbygoogle|‘|window.adsbygoogle]', '', news)

        words = news.split(" ")

        for i in range(0, len(words)):

            if words[i] in feature_dictionary:

                lenghts = feature_lengths.get(words[i])

                for lens in lenghts:

                    bgrm = words[i]

                    for j in range(i + 1, i + lens):

                        if j < len(words):

                            bgrm = bgrm + ' ' + words[j]
                        else:

                            break

                    if bgrm in feature_dictionary.get(words[i]):

                        if dataset[current][bigrams.index(bgrm)] == 0:
                            dataset[current][bigrams.index(bgrm)] = 1
        for i in range(0, len(words)):

            if words[i] in victim1:

                if dataset[current][1088] == 0:
                    dataset[current][1088] = 1
            elif words[i] in victim2:

                if dataset[current][1089] == 0:
                    dataset[current][1089] = 1

            elif i + 1 < len(words):

                vbgrm = words[i] + ' ' + words[i + 1]

                if vbgrm in wasVictim:

                    if dataset[current][1090] == 0:
                        dataset[current][1090] = 1


        classifier = pickle.load(open(os.path.join(settings.TEXT_ROOT,"classifier.txt"), 'rb'))
        label = classifier.predict(dataset)
        f = ''
        if label == 0 :
            f = 'No Violance Event Detected'
        if label == 1 :
            f = 'Kidnap Event Detected'
        if label == 2 :
            f = 'Adult Suicide Event Detected'
        if label == 3 :
            f = 'Clash Event Detected'
        if label == 4 :
            f = 'Teen Suicide Event Detected '
        if label == 5 :
            f = 'Murder Event Detected'
        if label == 6 :
            f = 'Rape Event Detected'
        return render(request,'text/newsresult.html', {"label":f , "news":news})
    else:
        return render(request,'text/index.html')

def about(request):
    return render(request, 'text/about.html')