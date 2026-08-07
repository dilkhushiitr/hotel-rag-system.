import 'package:intl/intl.dart';

extension CurrencyExtension on double {
  String get inr {
    final formatter = NumberFormat.currency(
      locale: 'en_IN',
      symbol: '₹',
      decimalDigits: 0,
    );
    return formatter.format(this);
  }

  String get inrDecimal {
    final formatter = NumberFormat.currency(
      locale: 'en_IN',
      symbol: '₹',
      decimalDigits: 2,
    );
    return formatter.format(this);
  }
}

extension DateExtension on DateTime {
  String get displayDate {
    final now = DateTime.now();
    if (year == now.year && month == now.month && day == now.day) {
      return 'Today, ${DateFormat('d MMM').format(this)}';
    }
    final yesterday = now.subtract(const Duration(days: 1));
    if (year == yesterday.year &&
        month == yesterday.month &&
        day == yesterday.day) {
      return 'Yesterday, ${DateFormat('d MMM').format(this)}';
    }
    return DateFormat('d MMM yyyy').format(this);
  }

  String get shortDate => DateFormat('d MMM').format(this);

  String get monthYear => DateFormat('MMMM yyyy').format(this);

  bool get isToday {
    final now = DateTime.now();
    return year == now.year && month == now.month && day == now.day;
  }

  bool get isThisMonth {
    final now = DateTime.now();
    return year == now.year && month == now.month;
  }

  bool get isThisWeek {
    final now = DateTime.now();
    final startOfWeek = now.subtract(Duration(days: now.weekday - 1));
    final start = DateTime(startOfWeek.year, startOfWeek.month, startOfWeek.day);
    return isAfter(start.subtract(const Duration(seconds: 1)));
  }
}

extension StringExtension on String {
  String get capitalize {
    if (isEmpty) return this;
    return '${this[0].toUpperCase()}${substring(1).toLowerCase()}';
  }

  String get titleCase {
    return split(' ').map((w) => w.capitalize).join(' ');
  }
}
