import 'package:animate_do/animate_do.dart';
import 'package:embelia/geminiAI/chat_bot_cubit.dart';
import 'package:embelia/routes/router.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:go_router/go_router.dart';

import '../authentication/user_auth.dart';
import '../constants.dart';
import '../widgets/feature_box.dart';
import '../widgets/pallets.dart';

class ChatBot extends StatefulWidget {
  const ChatBot({super.key});
  static const String id = 'ChatBot';

  @override
  State<ChatBot> createState() => _ChatBotState();
}

class _ChatBotState extends State<ChatBot> {
  final flutterTts = FlutterTts();
  String? generatedContent;
  String? generatedImageUrl;
  int start = 200;
  int delay = 200;

  @override
  void initState() {
    super.initState();
    BlocProvider.of<ChatBotCubit>(context).initSpeech();
    BlocProvider.of<ChatBotCubit>(context).getUserDataForGemini();
  }

  Future<void> systemSpeak(String content) async {
    await flutterTts.speak(content);
  }

  @override
  void dispose() {
    super.dispose();
    BlocProvider.of<ChatBotCubit>(context).stopListening();
    flutterTts.stop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: MyColor.primaryColor,
        title: BounceInDown(
          child: Text(
            'Hello ${UserAuth().userName.toString().toTitleCase()}',
            style: const TextStyle(
                fontFamily: 'OpenSans',
                fontWeight: FontWeight.bold,
                fontSize: 20),
          ),
        ),
        leading: IconButton(
          onPressed: () {
            GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen);
          },
          icon: const Icon(Icons.arrow_back),
        ),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // virtual assistant picture
            const SizedBox(
              height: 20,
            ),
            ZoomIn(
              child: Stack(
                children: [
                  SizedBox(
                    height: 123,
                    child: SvgPicture.asset(
                      'assets/images/Google-Ai-Gemini.svg',
                    ),
                  ),
                ],
              ),
            ),
            // chat bubble
            FadeInRight(
              child: Visibility(
                visible: generatedImageUrl == null,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 10,
                  ),
                  margin: const EdgeInsets.symmetric(horizontal: 40).copyWith(
                    top: 30,
                  ),
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: Pallete.borderColor,
                    ),
                    borderRadius: BorderRadius.circular(20).copyWith(
                      topLeft: Radius.zero,
                    ),
                  ),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 10.0),
                    child: BlocBuilder<ChatBotCubit, ChatBotState>(
                      builder: (context, state) {
                        if (state is ChatBotLoaded) {
                          generatedContent = state.message;
                          systemSpeak(generatedContent!);
                        } else if (state is ChatBotError) {
                          generatedContent = null;
                        } else if (state is ChatBotListening) {
                          generatedContent = 'Listening...';
                        } else if (state is ChatBotLoading) {
                          generatedContent =
                              'Generating Appropriate Response...';
                        } else {
                          if (DateTime.now().hour < 12) {
                            return Text(
                              generatedContent == null
                                  ? 'Good Morning, what task can I do for you?'
                                  : generatedContent!,
                              style: TextStyle(
                                fontFamily: 'Cera Pro',
                                color: Pallete.mainFontColor,
                                fontSize: generatedContent == null ? 25 : 18,
                              ),
                            );
                          } else if (DateTime.now().hour < 16) {
                            return Text(
                              generatedContent == null
                                  ? 'Good Afternoon, what task can I do for you?'
                                  : generatedContent!,
                              style: TextStyle(
                                fontFamily: 'Cera Pro',
                                color: Pallete.mainFontColor,
                                fontSize: generatedContent == null ? 25 : 18,
                              ),
                            );
                          } else {
                            return Text(
                              generatedContent == null
                                  ? 'Good Evening, what task can I do for you?'
                                  : generatedContent!,
                              style: TextStyle(
                                fontFamily: 'Cera Pro',
                                color: Pallete.mainFontColor,
                                fontSize: generatedContent == null ? 25 : 18,
                              ),
                            );
                          }
                        }
                        return Text(
                          generatedContent == null
                              ? 'Good Morning, what task can I do for you?'
                              : generatedContent!,
                          style: TextStyle(
                            fontFamily: 'Cera Pro',
                            color: Pallete.mainFontColor,
                            fontSize: generatedContent == null ? 25 : 18,
                          ),
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),
            if (generatedImageUrl != null)
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(20),
                  child: Image.network(generatedImageUrl!),
                ),
              ),
            SlideInLeft(
              child: Visibility(
                visible: generatedContent == null && generatedImageUrl == null,
                child: Container(
                  padding: const EdgeInsets.all(10),
                  alignment: Alignment.centerLeft,
                  margin: const EdgeInsets.only(top: 10, left: 22),
                  child: const Text(
                    'Here are a few features',
                    style: TextStyle(
                      fontFamily: 'OpenSans',
                      color: Pallete.mainFontColor,
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),
            // features list
            Visibility(
              visible: generatedContent == null && generatedImageUrl == null,
              child: Column(
                children: [
                  SlideInLeft(
                    delay: Duration(milliseconds: start),
                    child: const FeatureBox(
                      color: Pallete.firstSuggestionBoxColor,
                      headerText: 'Smart Assistant',
                      descriptionText:
                          'A smarter way to stay organized and informed with Gemini AI',
                    ),
                  ),
                  const SizedBox(height: 20),
                  SlideInLeft(
                    delay: Duration(milliseconds: start + delay),
                    child: Stack(
                      children: [
                        const FeatureBox(
                          color: Pallete.secondSuggestionBoxColor,
                          headerText: 'Image Generation',
                          descriptionText:
                              'Get inspired and stay creative with your personal assistant powered by Gemini AI',
                        ),
                        Positioned(
                          top: -15,
                          right: 10,
                          child: AnimatedContainer(
                            duration: const Duration(milliseconds: 500),
                            transform: Matrix4.rotationZ(0.5),
                            child: GestureDetector(
                              child: SvgPicture.asset(
                                'assets/images/premium.svg',
                                height: 45,
                              ),
                              onTap: () {
                                // Show that it is a premium feature as showDialog, and of minimum size
                                showDialog(
                                  context: context,
                                  builder: (context) {
                                    return AlertDialog(
                                      // design it as a faded box
                                      backgroundColor:
                                          Colors.white.withOpacity(0.8),

                                      title: const Text('Premium Feature'),
                                      content: const Text(
                                          'This feature is only available to premium users'),
                                      actions: [
                                        TextButton(
                                          onPressed: () {
                                            Navigator.of(context).pop();
                                          },
                                          child: const Text('Close'),
                                        ),
                                      ],
                                    );
                                  },
                                );
                              },
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 10),
                  SlideInLeft(
                    delay: Duration(milliseconds: start + 2 * delay),
                    child: const FeatureBox(
                      color: Pallete.thirdSuggestionBoxColor,
                      headerText: 'Smart Voice Assistant',
                      descriptionText:
                          'Get the best of both worlds with a voice assistant powered Gemini AI',
                    ),
                  ),
                ],
              ),
            )
          ],
        ),
      ),
      floatingActionButton: ZoomIn(
        delay: Duration(milliseconds: start + 3 * delay),
        child: FloatingActionButton(
          backgroundColor: Pallete.firstSuggestionBoxColor,
          onPressed: () {
            // If not yet listening for speech start, otherwise stop
            BlocProvider.of<ChatBotCubit>(context).isListening()
                ? BlocProvider.of<ChatBotCubit>(context).stopListening()
                : flutterTts.stop().then((value) =>
                    BlocProvider.of<ChatBotCubit>(context).startListening());
          },
          tooltip: 'Listen',
          child: BlocBuilder<ChatBotCubit, ChatBotState>(
            builder: (context, state) {
              if (state is ChatBotInitial) {
                return const Icon(
                  Icons.mic,
                  color: Colors.black,
                );
              } else if (state is ChatBotListening) {
                return const CircularProgressIndicator();
              } else if (state is ChatBotLoaded) {
                return const Icon(Icons.mic);
              } else if (state is ChatBotError) {
                return const Icon(Icons.mic_off);
              }
              return const Icon(Icons.mic);
            },
          ),
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        selectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.bold, fontFamily: 'OpenSans'),
        unselectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.bold, fontFamily: 'OpenSans'),
        landscapeLayout: BottomNavigationBarLandscapeLayout.linear,
        elevation: 10,
        currentIndex: 1,
        backgroundColor: MyColor.lightGreyShade, // Set the background color
        selectedItemColor: Colors
            .orangeAccent.shade400, // Color of selected item's icon and label
        unselectedItemColor:
            Colors.grey, // Color of unselected item's icon and label
        selectedFontSize: 14, // Font size of the selected item's label
        unselectedFontSize: 12, // Font size of unselected items' labels
        type: BottomNavigationBarType.fixed, // Fixed type for a stable layout
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: "Home",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.chat_rounded),
            label: "Chat",
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.question_answer),
            label: "Help",
            tooltip: "FAQ",
          ),
        ],
        onTap: (index) {
          if (index == 0) {
            GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen);
          }
          if (index == 1) {
            GoRouter.of(context).goNamed(MyAppRouteConstants.chatBotScreen);
          }
          if (index == 2) {
            GoRouter.of(context).goNamed(MyAppRouteConstants.kFAQ);
          }
        },
      ),
    );
  }
}
