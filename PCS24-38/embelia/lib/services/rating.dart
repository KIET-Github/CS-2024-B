import 'package:flutter/material.dart';
import 'package:flutter_rating_bar/flutter_rating_bar.dart';

class Rating extends StatefulWidget {
  Rating(
      {Key? key,
      required this.maxRating,
      required this.initialRating,
      required this.onRatingUpdate})
      : super(key: key);
  final int maxRating;
  final double initialRating;
  void Function(double) onRatingUpdate;


  @override
  State<Rating> createState() => _RatingState();
}

class _RatingState extends State<Rating> {
  @override
  Widget build(BuildContext context) {
    return RatingBar(
      itemPadding: const EdgeInsets.symmetric(horizontal: 4.0),
      direction: Axis.horizontal,
      glowRadius: 1.0,
      itemCount: widget.maxRating,
      initialRating: widget.initialRating,
      ratingWidget: RatingWidget(
        full: const Icon(Icons.star, color: Colors.amber, size: 10,),
        half: const Icon(Icons.star_half, color: Colors.yellow, size: 10,),
        empty: const Icon(Icons.star_border, color: Colors.yellow, size: 10,),
      ),
      onRatingUpdate: widget.onRatingUpdate,
    );
  }
}