import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';

kSnackBarNotify(BuildContext context, String message) {
  var snackBar = SnackBar(
    duration: const Duration(seconds: 1),
    backgroundColor: const Color.fromRGBO(200, 100, 50, 0.8),
    content: Text(message),
  );
  ScaffoldMessenger.of(context).showSnackBar(snackBar);
}

// class AnimationMaker extends StatefulWidget {
//   @override
//   _AnimationMakerState createState() => _AnimationMakerState();
// }

// class _AnimationMakerState extends State<AnimationMaker> with SingleTickerProviderStateMixin {
//   late AnimationController _controller;
//   bool _isSelected = false;

//   @override
//   void initState() {
//     super.initState();
//     _controller = AnimationController(
//         vsync: this, duration: const Duration(milliseconds: 2000));
//   }

//   @override
//   Widget build(BuildContext context) {
//     return SizedBox(
//       height: 100,
//       width: 100,
//       child: GestureDetector(
//         onTap: _handleTap,
//         child: Lottie.network(
//           frameRate: FrameRate(60),
//           lotties,
//           controller: _controller,
//         ),
//       ),
//     );
//   }

//   void _handleTap() {
//     _isSelected ? _controller.reverse() : _controller.forward();
//     setState(() {
//       _isSelected = !_isSelected;
//     });
//   }
// }
