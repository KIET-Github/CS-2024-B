import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:go_router/go_router.dart';

import '../../constants.dart';
import '../../routes/router.dart';
import 'faq_provider_cubit.dart';

class FAQ extends StatefulWidget {
  static const String id = "FAQ";
  const FAQ({Key? key}) : super(key: key);

  @override
  State<FAQ> createState() => _FAQState();
}

class _FAQState extends State<FAQ> {
  final TextEditingController _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        leading: IconButton(
          onPressed: () =>
              GoRouter.of(context).goNamed(MyAppRouteConstants.homeScreen),
          icon: const Icon(Icons.arrow_back),
        ),
        backgroundColor: MyColor.primaryColor,
        title: const Text(
          "FAQ",
          style: TextStyle(fontWeight: FontWeight.bold, fontFamily: 'OpenSans'),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(8.0),
        child: SingleChildScrollView(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(10.0),
                child: Row(
                  children: [
                    SizedBox(
                      width: MediaQuery.of(context).size.width * 0.7,
                      child: TextFormField(
                        controller: _controller,
                        decoration: const InputDecoration(
                          hintText: "Write your query here ...",
                          labelText: "Query",
                          border: OutlineInputBorder(
                            borderSide:
                                BorderSide(color: Colors.black, width: 2),
                            borderRadius: BorderRadius.all(
                              Radius.circular(10),
                            ),
                          ),
                        ),
                      ),
                    ),
                    const Spacer(),
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: MyColor.lightGreyShade,
                        shape: const RoundedRectangleBorder(
                          borderRadius: BorderRadius.all(
                            Radius.circular(15),
                          ),
                        ),
                        side: const BorderSide(color: Colors.black, width: 1),
                      ),
                      onPressed: () async {
                        if (_controller.text.isEmpty) {
                          return;
                        }
                        await BlocProvider.of<FaqProviderCubit>(context)
                            .getFAQAnswer(_controller.text);
                      },
                      child: const Icon(Icons.search),
                    ),
                  ],
                ),
              ),
              Card(
                color: MyColor.lightGreyShade,
                shape: const RoundedRectangleBorder(
                  side: BorderSide(color: Colors.black, width: 1.5),
                  borderRadius: BorderRadius.all(
                    Radius.circular(15),
                  ),
                ),
                child: BlocConsumer<FaqProviderCubit, FaqProviderState>(
                  listener: (BuildContext context, state) {},
                  builder: (BuildContext context, FaqProviderState state) {
                    if (state is FaqProviderLoading) {
                      return const ListTile(
                        title: Align(
                          alignment: Alignment.center,
                          child: CircularProgressIndicator(
                            color: Colors.black,
                          ),
                        ),
                      );
                    } else if (state is FaqProviderLoaded) {
                      return ListTile(
                        title: Align(
                          alignment: Alignment.center,
                          child: Text(
                            BlocProvider.of<FaqProviderCubit>(context).answer,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontFamily: 'OpenSans',
                            ),
                          ),
                        ),
                      );
                    } else {
                      return const ListTile(
                        title: Align(
                          alignment: Alignment.center,
                          child: Text(
                            "Mental Health",
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontFamily: 'OpenSans',
                            ),
                          ),
                        ),
                      );
                    }
                  },
                ),
              ),
            ],
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
        currentIndex: 2,
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
