import 'dart:math';

import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:embelia/authentication/cubit/user_data_firebase_cubit.dart';
import 'package:embelia/constants.dart';
import 'package:embelia/geminiAI/chat_bot_cubit.dart';
import 'package:embelia/routes/router.dart';
import 'package:embelia/screens/homeScreen/auth_cubit.dart';
import 'package:embelia/screens/homeScreen/home_screen_cubit.dart';
import 'package:embelia/server/mongoDB.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:flutter_svg/svg.dart';
import 'package:go_router/go_router.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';

import '../../authentication/user_auth.dart';
import '../../server/user_task_model.dart';
import '../../utils/notification_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  NotificationServices notificationServices = NotificationServices();
  final TextEditingController _taskNameController = TextEditingController();
  final TextEditingController _taskTimeController = TextEditingController();

  @override
  void initState() {
    getRecommendedTaskAndAddToFirebase(context);
    notificationServices.initialiseNotification();
    BlocProvider.of<UserDataFirebaseCubit>(context).init(context).then(
          (value) => BlocProvider.of<UserDataFirebaseCubit>(context)
              .doesUserHealthProfileExist(context)
              .then(
            (value) {
              if (value == false) {
                GoRouter.of(context)
                    .goNamed(MyAppRouteConstants.userHeathProfile);
              }
            },
          ),
        );
    UserAuth().getUserName();
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      endDrawer: Drawer(
        width: MediaQuery.of(context).size.width / 1.5,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            const SizedBox(
              height: 50,
            ),
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 15.0, vertical: 10.0),
              child: Container(
                decoration: BoxDecoration(
                  // color: MyColor.lightGreyShade,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: MaterialBanner(
                    dividerColor: Colors.green,
                    surfaceTintColor: Colors.green.shade100,
                    contentTextStyle: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontFamily: 'Ubuntu',
                      color: Colors.black,
                    ),
                    backgroundColor: Colors.green.shade100,
                    leading: const Icon(
                      Icons.verified,
                      color: Colors.green,
                    ),
                    content: const Text(
                      'Your account is verified!',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontFamily: 'Ubuntu',
                      ),
                    ),
                    actions: [
                      IconButton(
                        onPressed: () {
                          showAdaptiveDialog(
                            context: context,
                            builder: (_) {
                              return AlertDialog(
                                title: const Text(
                                  'Account Verification',
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontFamily: 'Ubuntu'),
                                ),
                                content: const Text(
                                  'Your account is verified!',
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontFamily: 'Ubuntu',
                                      color: Colors.green,
                                      fontSize: 15),
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () {
                                      Navigator.of(context).pop();
                                    },
                                    child: const Text('OK'),
                                  ),
                                ],
                              );
                            },
                          );
                        },
                        icon: const Icon(
                          Icons.info,
                          color: Colors.black,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            Divider(
              thickness: 2,
              endIndent: 20,
              indent: 20,
              color: Colors.black.withOpacity(0.5),
            ),
            const Spacer(),
            const SizedBox(
              height: 10,
            ),
            Divider(
              thickness: 2,
              endIndent: 20,
              indent: 20,
              color: Colors.black.withOpacity(0.5),
            ),
            Text(
              'Water Reminder?',
              style: GoogleFonts.ubuntu(
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
            const SizedBox(
              height: 10,
            ),
            ToggleButtons(
              onPressed: (int index) async {
                if (index == 0) {
                  await notificationServices.scheduleNotification(
                      'Reminder', 'Please have a glass of water');
                } else {
                  await notificationServices.cancelNotification();
                }
                setState(() {});
              },
              borderRadius: BorderRadius.circular(10),
              textStyle: const TextStyle(
                fontWeight: FontWeight.bold,
              ),
              borderColor: Colors.grey,
              fillColor: Colors.grey.shade200,
              selectedColor: Colors.green,
              disabledColor: Colors.grey,
              color: Colors.black,
              isSelected: const [true, false],
              children: const [
                Icon(Icons.alarm_on_sharp),
                Icon(Icons.alarm_off_sharp),
              ],
            ),
            const SizedBox(
              height: 10,
            ),
            Divider(
              thickness: 2,
              endIndent: 20,
              indent: 20,
              color: Colors.black.withOpacity(0.5),
            ),
            const SizedBox(
              height: 10,
            ),
            Row(
              children: [
                const Spacer(),
                Text(
                  'Logged in as',
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.ubuntu(
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                ),
                const SizedBox(
                  width: 5,
                ),
                Text(
                  UserAuth().userName.toString().toTitleCase(),
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.ubuntu(
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                    color: Colors.blue,
                  ),
                ),
                const Spacer(),
              ],
            ),
            const SizedBox(
              height: 10,
            ),
            Divider(
              thickness: 2,
              endIndent: 20,
              indent: 20,
              color: Colors.black.withOpacity(0.5),
            ),
            const SizedBox(
              height: 10,
            ),
            Stack(
              alignment: Alignment.center,
              children: [
                Align(
                  alignment: Alignment.center,
                  child: Container(
                    height: 50,
                    width: 50,
                    decoration: BoxDecoration(
                      color: MyColor.lightGreyShade,
                      borderRadius: BorderRadius.circular(50),
                    ),
                  ),
                ),
                BlocBuilder<AuthCubit, AuthState>(
                  builder: (context, state) {
                    if (state is AuthInitial) {
                      return IconButton(
                        onPressed: () async {
                          BlocProvider.of<AuthCubit>(context).signOut(context);
                        },
                        icon: const Icon(
                          CupertinoIcons.square_arrow_left,
                          color: Colors.black,
                          size: 25,
                        ),
                      );
                    } else if (state is AuthLoading) {
                      return const Center(
                        child: CircularProgressIndicator(),
                      );
                    } else {
                      return IconButton(
                        onPressed: () async {},
                        icon: const Icon(
                          CupertinoIcons.square_arrow_left,
                          color: Colors.black,
                          size: 25,
                        ),
                      );
                    }
                  },
                ),
              ],
            ),
            Text(
              'Sign Out',
              style: GoogleFonts.ubuntu(
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
            const SizedBox(
              height: 20,
            ),
          ],
        ),
      ),
      appBar: AppBar(
        title: const Text(
          "Home",
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: MyColor.primaryColor,
        centerTitle: true,
      ),
      body: StreamBuilder<QuerySnapshot>(
        stream: FirebaseFirestore.instance.collection('users').snapshots(),
        builder: (BuildContext context, AsyncSnapshot<QuerySnapshot> snapshot) {
          if (snapshot.hasError) {
            return const Text('Something went wrong');
          }
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }

          return ListView(
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20.0, vertical: 10),
                    child: Container(
                      decoration: BoxDecoration(
                        color: MyColor.lightGreyShade,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Row(
                          children: [
                            Icon(
                              Icons.fireplace_rounded,
                              color: Colors.red[700],
                              size: 30,
                            ),
                            const SizedBox(
                              width: 10,
                            ),
                            Align(
                              alignment: Alignment.centerLeft,
                              child: BlocBuilder<UserDataFirebaseCubit,
                                  UserDataFirebaseState>(
                                builder: (context, state) {
                                  if (state is UserDataFirebaseLoaded) {
                                    return Text(
                                      state.streak.toString(),
                                    );
                                  } else {
                                    return const Text('0');
                                  }
                                },
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20.0, vertical: 10),
                    child: Container(
                      decoration: BoxDecoration(
                        color: MyColor.lightGreyShade,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: Row(
                          children: [
                            Icon(
                              Icons.monetization_on,
                              color: Colors.amber[700],
                              size: 30,
                            ),
                            const SizedBox(
                              width: 10,
                            ),
                            Align(
                              alignment: Alignment.centerLeft,
                              child: BlocBuilder<UserDataFirebaseCubit,
                                  UserDataFirebaseState>(
                                builder: (context, state) {
                                  if (state is UserDataFirebaseLoaded) {
                                    return Text(
                                      state.prizeMoney.toString(),
                                    );
                                  } else if (state is UserDataFirebaseLoading) {
                                    return const Center(
                                        child: CircularProgressIndicator());
                                  } else {
                                    return const Text('0');
                                  }
                                },
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 20.0, vertical: 2),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child: Text("Welcome to Embelia"),
                ),
              ),
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20.0, vertical: 2),
                child: Align(
                  alignment: Alignment.centerLeft,
                  child:
                      BlocBuilder<UserDataFirebaseCubit, UserDataFirebaseState>(
                    builder: (context, state) {
                      if (state is UserDataFirebaseLoaded) {
                        return Text(
                          "Hey ${UserAuth().userName.toString().toTitleCase()}",
                          style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                              fontFamily: 'Ubuntu'),
                        );
                      } else if (state is UserDataFirebaseLoading) {
                        return Center(
                            child: CircularProgressIndicator(
                          color: Color.fromRGBO(
                              Random().nextInt(255),
                              Random().nextInt(255),
                              Random().nextInt(255),
                              0.8),
                        ));
                      } else {
                        return const Text('Hey User');
                      }
                    },
                  ),
                ),
              ),
              const SizedBox(
                height: 40,
              ),
              BlocBuilder<UserDataFirebaseCubit, UserDataFirebaseState>(
                builder: (context, state) {
                  if (state is UserDataFirebaseLoaded) {
                    return CircularPercentIndicator(
                      animateFromLastPercent: true,
                      header: Text(
                        "Health Score",
                        style: TextStyle(
                            color: Colors.grey.shade600,
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'OpenSans'),
                      ),
                      footer: SvgPicture.asset(
                        'assets/images/health-heart.svg',
                        height: 50,
                        width: 50,
                      ),
                      lineWidth: 10,
                      center: Text(
                        state.healthScore.toString(),
                        style: const TextStyle(
                            fontSize: 30,
                            fontWeight: FontWeight.bold,
                            fontFamily: 'Ubuntu'),
                      ),
                      radius: 90,
                      progressColor: _progressColor(state.healthScore),
                      percent: state.healthScore / 100,
                      widgetIndicator: SvgPicture.asset(
                        'assets/images/health-heart.svg',
                        height: 50,
                        width: 50,
                      ),
                    );
                  } else if (state is UserDataFirebaseLoading) {
                    return const Center(child: CircularProgressIndicator());
                  } else {
                    return const Text('0');
                  }
                },
              ),
              const SizedBox(
                height: 20,
              ),
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20.0, vertical: 2),
                child: BlocBuilder<HomeScreenCubit, HomeScreenState>(
                  builder: (context, state) {
                    return Row(
                      children: [
                        GestureDetector(
                          onTap: () {
                            if (state is HomeScreenRecommendedTask) {
                              BlocProvider.of<HomeScreenCubit>(context)
                                  .changeTask();
                            }
                          },
                          child: Container(
                            decoration: BoxDecoration(
                              color: state is HomeScreenInitialTask
                                  ? const Color(0xffB8BDC4)
                                  : MyColor.lightGreyShade,
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(8.0),
                              child: Row(
                                children: [
                                  const Align(
                                    alignment: Alignment.centerLeft,
                                    child: Text(
                                      "Your Tasks!",
                                      style: TextStyle(
                                        fontFamily: 'Ubuntu',
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(
                                    width: 5,
                                  ),
                                  SvgPicture.asset(
                                    'assets/images/fail.svg',
                                    height: 25,
                                    width: 50,
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                        const Spacer(),
                        GestureDetector(
                          onTap: () {
                            if (state is HomeScreenInitialTask) {
                              BlocProvider.of<HomeScreenCubit>(context)
                                  .changeTask();
                            }
                          },
                          child: Container(
                            decoration: BoxDecoration(
                              color: state is HomeScreenRecommendedTask
                                  ? const Color(0xffB8BDC4)
                                  : MyColor.lightGreyShade,
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Padding(
                              padding: const EdgeInsets.all(8.0),
                              child: Row(
                                children: [
                                  const Align(
                                    alignment: Alignment.centerLeft,
                                    child: Text(
                                      "Recommended Tasks!",
                                      style: TextStyle(
                                        fontFamily: 'Ubuntu',
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ),
                                  const SizedBox(
                                    width: 5,
                                  ),
                                  SvgPicture.asset(
                                    'assets/images/fail.svg',
                                    height: 25,
                                    width: 50,
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),
              BlocBuilder<HomeScreenCubit, HomeScreenState>(
                builder: (context, state) {
                  if (state is HomeScreenInitialTask) {
                    return Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Container(
                        decoration: BoxDecoration(
                          color: const Color(0xffE6EDf5),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        height: MediaQuery.of(context).size.height / 2.5,
                        child: BlocBuilder<UserDataFirebaseCubit,
                            UserDataFirebaseState>(
                          builder: (context, state) {
                            if (state is UserDataFirebaseLoaded) {
                              return StreamBuilder<QuerySnapshot>(
                                stream: FirebaseFirestore.instance
                                    .collection('users')
                                    .doc(UserAuth().userEmail)
                                    .collection('totalTask')
                                    .snapshots(),
                                builder: (BuildContext context,
                                    AsyncSnapshot<QuerySnapshot> snapshot) {
                                  if (snapshot.hasError) {
                                    return const Text('Something went wrong');
                                  }
                                  if (snapshot.connectionState ==
                                      ConnectionState.waiting) {
                                    return const Center(
                                        child: CircularProgressIndicator());
                                  }

                                  return ListView(
                                    children: snapshot.data!.docs.map(
                                      (DocumentSnapshot document) {
                                        Map<String, dynamic> data = document
                                            .data()! as Map<String, dynamic>;

                                        double remainingTime = ((DateTime.now()
                                                    .millisecondsSinceEpoch -
                                                data['taskStartTime']) /
                                            3600000);
                                        remainingTime = data['taskTotalTime'] -
                                            remainingTime;
                                        return Padding(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 15.0, vertical: 10),
                                          child: CheckboxListTile(
                                            checkboxShape:
                                                RoundedRectangleBorder(
                                              borderRadius:
                                                  BorderRadius.circular(10),
                                            ),
                                            shape: RoundedRectangleBorder(
                                              borderRadius:
                                                  BorderRadius.circular(10),
                                            ),
                                            activeColor: MyColor.primaryColor,
                                            tileColor: MyColor.lightGreyShade,
                                            value: false,
                                            onChanged: (value) {
                                              // Set Data in Cubit
                                              BlocProvider.of<
                                                          UserDataFirebaseCubit>(
                                                      context)
                                                  .setTaskComplete(value!);

                                              // Delete Task Data
                                              final taskName = data['taskName'];
                                              // final taskData = DataSchema(
                                              //   name: UserAuth.userName!,
                                              //   email: UserAuth.userEmail,
                                              //   taskName: state.taskName,
                                              //   taskStartTime: state.taskStartTime,
                                              //   taskTotalTime:
                                              //       data['taskTotalTime'] as int,
                                              // );

                                              // Delete from Firebase
                                              BlocProvider.of<
                                                          UserDataFirebaseCubit>(
                                                      context)
                                                  .deleteUserTaskData(
                                                      context, taskName);

                                              // Delete from MongoDB
                                              // MongoDB.deleteOne(
                                              //     taskData.toMap(), UserAuth.userEmail);
                                            },
                                            title: Text(
                                              data['taskName'].toString(),
                                              style: const TextStyle(
                                                  fontFamily: 'Ubuntu',
                                                  fontStyle: FontStyle.italic,
                                                  fontWeight: FontWeight.bold),
                                            ),
                                            subtitle: Row(
                                              children: [
                                                Text(
                                                  "Exp. Time: ${data['taskTotalTime'].toString()} hrs",
                                                  style: const TextStyle(
                                                    color: Colors.black45,
                                                    fontWeight: FontWeight.bold,
                                                    fontFamily: 'Ubuntu',
                                                  ),
                                                ),
                                                const Text(
                                                  " | ",
                                                  style: TextStyle(
                                                    color: Colors.blue,
                                                    fontWeight: FontWeight.bold,
                                                  ),
                                                ),
                                                Text(
                                                  "Rem Time: ${remainingTime.toStringAsFixed(3)} hrs",
                                                  style: TextStyle(
                                                    color: remainingTime <
                                                            (data['taskTotalTime'] *
                                                                0.2)
                                                        ? Colors.red
                                                        : Colors.green,
                                                    fontWeight: FontWeight.bold,
                                                    fontFamily: 'Ubuntu',
                                                  ),
                                                ),
                                              ],
                                            ),
                                          ),
                                        );
                                      },
                                    ).toList(),
                                  );
                                },
                              );
                            } else if (state is UserDataFirebaseLoading) {
                              return const Center(
                                child: CircularProgressIndicator(),
                              );
                            } else {
                              return Text('No Tasks',
                                  style: GoogleFonts.ubuntu());
                            }
                          },
                        ),
                      ),
                    );
                  } else {
                    return Padding(
                      padding: const EdgeInsets.all(5.0),
                      child: Container(
                        decoration: BoxDecoration(
                          color: const Color(0xffE6EDf5),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        height: MediaQuery.of(context).size.height / 2.5,
                        child: BlocBuilder<UserDataFirebaseCubit,
                            UserDataFirebaseState>(
                          builder: (context, state) {
                            if (state is UserDataFirebaseLoaded) {
                              return StreamBuilder<QuerySnapshot>(
                                stream: FirebaseFirestore.instance
                                    .collection('recommendedTasks')
                                    .snapshots(),
                                builder: (BuildContext context,
                                    AsyncSnapshot<QuerySnapshot> snapshot) {
                                  if (snapshot.hasError) {
                                    return const Text('Something went wrong');
                                  }
                                  if (snapshot.connectionState ==
                                      ConnectionState.waiting) {
                                    return const Center(
                                        child: CircularProgressIndicator());
                                  }
                                  return ListView(
                                    children: snapshot.data!.docs.map(
                                      (DocumentSnapshot document) {
                                        Map<String, dynamic> data = document
                                            .data()! as Map<String, dynamic>;
                                        return Padding(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 15.0, vertical: 10),
                                          child: ListTile(
                                            shape: RoundedRectangleBorder(
                                              borderRadius:
                                                  BorderRadius.circular(10),
                                            ),
                                            tileColor: MyColor.lightGreyShade,
                                            onTap: () {},
                                            title: Text(
                                              data['taskName'].toString(),
                                              style: const TextStyle(
                                                  fontFamily: 'Ubuntu',
                                                  fontStyle: FontStyle.italic,
                                                  fontWeight: FontWeight.bold),
                                            ),
                                            subtitle: Text(
                                              "Exp. Time: ${data['taskTotalTime'].toString()} hrs",
                                              style: const TextStyle(
                                                color: Colors.black45,
                                                fontWeight: FontWeight.bold,
                                                fontFamily: 'Ubuntu',
                                              ),
                                            ),
                                            trailing: IconButton(
                                              onPressed: () {
                                                // Set Data in Cubit
                                                BlocProvider.of<
                                                            UserDataFirebaseCubit>(
                                                        context)
                                                    .setTaskName(
                                                        data['taskName']
                                                            .toString());

                                                // Create Task Data
                                                final taskData = DataSchema(
                                                  name: UserAuth().userName!,
                                                  email: UserAuth().userEmail,
                                                  taskName: data['taskName']
                                                      .toString(),
                                                  taskStartTime: DateTime.now()
                                                      .millisecondsSinceEpoch,
                                                  taskTotalTime:
                                                      data['taskTotalTime']
                                                          .toDouble(),
                                                );

                                                // Add to Firebase
                                                BlocProvider.of<
                                                            UserDataFirebaseCubit>(
                                                        context)
                                                    .addUserTaskData(context,
                                                        taskData.toMap());

                                                // Add to MongoDB
                                                MongoDB.insertOne(
                                                    taskData.toMap(),
                                                    UserAuth().userEmail);
                                              },
                                              icon: SvgPicture.asset(
                                                'assets/images/add.svg',
                                                height: 30,
                                                width: 30,
                                              ),
                                            ),
                                          ),
                                        );
                                      },
                                    ).toList(),
                                  );
                                },
                              );
                            } else if (state is UserDataFirebaseLoading) {
                              return const Center(
                                child: CircularProgressIndicator(),
                              );
                            } else {
                              return Text('No Tasks',
                                  style: GoogleFonts.ubuntu());
                            }
                          },
                        ),
                      ),
                    );
                  }
                },
              ),
            ],
          );
        },
      ),
      bottomNavigationBar: BottomNavigationBar(
        selectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.bold, fontFamily: 'OpenSans'),
        unselectedLabelStyle: const TextStyle(
            fontWeight: FontWeight.bold, fontFamily: 'OpenSans'),
        landscapeLayout: BottomNavigationBarLandscapeLayout.linear,
        elevation: 10,
        currentIndex: 0,
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
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          showDialog(
            context: context,
            builder: (context) {
              return AlertDialog(
                title: Text(
                  'Add Task',
                  textAlign: TextAlign.center,
                  style: GoogleFonts.openSans(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                content: SizedBox(
                  height: 130,
                  width: 150,
                  child: Column(
                    children: [
                      TextField(
                        controller: _taskNameController,
                        keyboardType: TextInputType.name,
                        decoration: const InputDecoration(
                          suffix: Icon(Icons.task),
                          hintText: 'Task Name',
                        ),
                      ),
                      TextField(
                        controller: _taskTimeController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          suffix: Text('hrs'),
                          hintText: 'Time',
                        ),
                      ),
                    ],
                  ),
                ),
                actions: [
                  TextButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    child: const Text('Cancel'),
                  ),
                  TextButton(
                    onPressed: () async {
                      // Set Data in Cubit
                      BlocProvider.of<UserDataFirebaseCubit>(context)
                          .setTaskName(_taskNameController.text);

                      // Create Task Data
                      final taskData = DataSchema(
                        name: UserAuth().userName!,
                        email: UserAuth().userEmail,
                        taskName: _taskNameController.text,
                        taskStartTime: DateTime.now().millisecondsSinceEpoch,
                        taskTotalTime: double.parse(_taskTimeController.text),
                      );

                      // Add to Firebase
                      BlocProvider.of<UserDataFirebaseCubit>(context)
                          .addUserTaskData(context, taskData.toMap());

                      // Add to MongoDB
                      MongoDB.insertOne(taskData.toMap(), UserAuth().userEmail);

                      // Close Dialog
                      Navigator.pop(context);
                    },
                    child: const Text('Add'),
                  ),
                ],
              );
            },
          );
        },
        child: SvgPicture.asset(
          'assets/images/add.svg',
          height: 30,
          width: 30,
        ),
      ),
    );
  }
}

Color _progressColor(int value) {
  if (value < 30) {
    return Colors.red;
  } else if (value < 40) {
    return Colors.orange;
  } else if (value < 55) {
    return Colors.blueGrey;
  } else if (value < 65) {
    return Colors.lightGreen;
  } else if (value < 90) {
    return Colors.green;
  } else {
    return Colors.green.shade900;
  }
}

Future<void> getRecommendedTaskAndAddToFirebase(BuildContext context) async {
  final tasks = await BlocProvider.of<ChatBotCubit>(context)
      .getRecommendationTasks(context);
  Future.delayed(const Duration(seconds: 5), () async {
    debugPrint(tasks.toString());
    debugPrint(tasks['taskName0']);
    for (int i = 0; i < tasks.length; i++) {
      await addRecommendedTask(
          tasks['taskName$i'].toString(), tasks['taskLength$i'] * 1.0, context);
    }
  });
}

Future<void> addRecommendedTask(
    String taskName, double taskTotalTime, BuildContext context) async {
  final taskData = DataSchema(
    name: UserAuth().userName!,
    email: UserAuth().userEmail,
    taskName: taskName,
    taskStartTime: DateTime.now().millisecondsSinceEpoch,
    taskTotalTime: taskTotalTime.toDouble(),
  );

  // Add to Firebase
  await BlocProvider.of<UserDataFirebaseCubit>(context)
      .addUserTaskData(context, taskData.toMap());

  // Add to MongoDB
  MongoDB.insertOne(taskData.toMap(), UserAuth().userEmail);
}
